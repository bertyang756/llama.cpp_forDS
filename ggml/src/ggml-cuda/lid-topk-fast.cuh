// LID Top-K fast path (O1): parallel 2-digit radix-select top-k.
// Ported from lid_topk_op/topk_reduce_v6.cuh (candidate D6), CAPF=2048, P=16384.
// P=16384: 超长上下文 prefill(n_lid≥32768) 三卡更快(CMP 15~35%); decode(nrows==1) 走 cub, P 不影响。
//
// 趟1(2 趟 8bit digit, MSB->LSB): 求每行第 k 大 key 的高 16 位阈值 T16(prefix 低 16 位恒 0)。
//   tkr_hist   : 每 block 覆盖 P 位置, 只计 (key & hi_mask)==(prefix & hi_mask) 的 digit 频次, 写局部直方图。
//   tkr_digit  : 每行 1 block, 累加 bpr 块直方图, 定 digit, 更新 prefix[row]/rem[row]。
// 趟2(tkr_gather): 每 block gather `(key & 0xffff0000) >= T16` 到 per-row gathered(gcount 原子计数, CAPF/行)。
//   gathered ⊇ top-k 集(正确性下界); gcount[row] 是精确命中数, 只有写入被 CAPF 截断。
// 趟3(tkr_final): 每行 1 block bitonic 排序 gathered(<=CAPF, 固定共享 CAPF*8=16KB)取前 k 写 dst。
//   gcount[row] > CAPF(超大 tie, 缓冲被截断)的行跳过, 交给趟4。
// 趟4(tkr_rowsel): 仅 overflow 行, 每行 1 block 对 64-bit 复合键 (f2k(v)<<32 | idx) 做 8 趟 8bit radix-select
//   求第 k 大阈值再 gather 恰好 k 个。复合键因 idx 唯一而全局唯一 => count(>=T)==k, 无 tie 歧义, 结果确定(顺序任意)。
//   ⚠️ 决策全在 device 上做: 绝不 host 读回/同步 —— 否则在 CUDA Graph capture 期间非法
//   (cudaStreamSynchronize/cudaMemcpyAsync D2H 会报 "operation not permitted when stream is capturing")。
//
// 语义：(值大优先, 值相同索引大优先)，每行前 k 个索引，行内 rank 0..k-1 值从大到小。
// 正确性：T16 是第 k 大 key 高 16 位下界 -> 所有 top-k 的 key 高16位>=T16 都被 gather;
//   gathered ⊇ top-k, final 取 gathered 的 top-k 即全局 top-k。overflow(超大 tie)回退 stock cub(正确 set)。
// 性能：2-digit 读从 4 趟减到 2 趟; CAPF=2048 全档 headroom 足(196608 档 gathered max≈1044); P=16384 大 ncols 优。
// 图安全：全链无 host 同步、无 D2H 拷贝, 可在 cudaStreamBeginCapture 区内发射(供 llama.cpp CUDA Graph 复用)。

#ifndef GGML_CUDA_LID_TOP_K_FAST_H
#define GGML_CUDA_LID_TOP_K_FAST_H

#include "common.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace ggml_lid_topk_fast {

constexpr int BLOCK = 256;   // 每 block 线程数
constexpr int P     = 16384; // 每 block 覆盖位置数 (超长上下文 prefill 最优; nrows==1 decode 走 cub 不经过此路径)
constexpr int CAPF  = 2048;  // 每行 gathered 缓冲/最终共享上限 (CAPF*8 = 16KB, 无需 opt-in)

struct Pair { float v; int i; };

__device__ __forceinline__ bool tkr_better(const Pair a, const Pair b) {
    return (a.v > b.v || (a.v == b.v && a.i > b.i));
}

// float -> uint32 key, key 升序 == float 升序(含 -INF/+INF 单调)。
__device__ __forceinline__ unsigned f2k(float f) {
    const unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? (~b) : (b | 0x80000000u);
}

__device__ __forceinline__ void bitonic_sort_pairs(
        float * s_val, int * s_idx, const int n, const int tid) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int e = tid; e < n; e += BLOCK) {
                const int l = e ^ j;
                if (l > e) {
                    const bool ascending = (e & k) == 0;
                    const Pair a = {s_val[e], s_idx[e]};
                    const Pair b = {s_val[l], s_idx[l]};
                    const bool swap = ascending ? tkr_better(a, b) : tkr_better(b, a);
                    if (swap) { s_val[e]=b.v; s_idx[e]=b.i; s_val[l]=a.v; s_idx[l]=a.i; }
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ unsigned tkr_himask(int pos) {
    return (pos == 24) ? 0u : (pos == 16) ? 0xff000000u : (pos == 8) ? 0xffff0000u : 0xffffff00u;
}

// 趟1.1: 每 block 局部直方图(当前 digit, 只计匹配 prefix 的 key)。grid=(bpr,nrows)。写 l_hist[gid*256+bin]。
__global__ void tkr_hist_kernel(
        const float * __restrict__ src,
        int * __restrict__ l_hist,
        const unsigned * __restrict__ prefix,
        const int ncols, const int bpr, const int pos) {
    const int b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const int gid = row * bpr + b;
    const float * rs = src + (size_t)row * ncols;
    const int cs = b * P, ce = min(cs + P, ncols);
    const unsigned pre = prefix[row];
    const unsigned hi_mask = tkr_himask(pos);

    __shared__ int sh[256];
    for (int i = t; i < 256; i++) sh[i] = 0;
    __syncthreads();
    for (int idx = cs + t; idx < ce; idx += BLOCK) {
        const unsigned key = f2k(rs[idx]);
        if ((key & hi_mask) == (pre & hi_mask))
            atomicAdd(&sh[(key >> pos) & 0xff], 1);
    }
    __syncthreads();
    int * out = l_hist + (size_t)gid * 256;
    for (int i = t; i < 256; i++) out[i] = sh[i];
}

// 趟1.2: 每行 1 block, 累加 bpr 块直方图, 从高位 cum 定当前 digit, 更新 prefix[row]/rem[row]。
__global__ void tkr_digit_kernel(
        const int * __restrict__ l_hist,
        unsigned * __restrict__ prefix, int * __restrict__ rem,
        const int nrows, const int bpr, const int pos) {
    const int row = blockIdx.x, t = threadIdx.x;
    __shared__ int s_hist[256];
    int sum = 0;
    for (int b = 0; b < bpr; b++) sum += l_hist[(size_t)(row * bpr + b) * 256 + t];
    s_hist[t] = sum;
    __syncthreads();
    if (t == 0) {
        unsigned pre = prefix[row];
        int r = rem[row];
        const unsigned hi_mask = tkr_himask(pos);
        int cum = 0, group = 0;
        for (int d = 255; d >= 0; d--) {
            cum += s_hist[d];
            if (cum >= r) { group = d; r -= (cum - s_hist[d]); break; }
        }
        prefix[row] = pre | ((unsigned)group << pos);
        rem[row] = r;
    }
}

// 趟2: 每 block gather 高16位>=T16 到 per-row gathered(按行原子计数)。grid=(bpr,nrows)。
// gathered 布局 CAPF/行(固定)。gcount[row] 精确记录命中总数, 超过 CAPF 时只有超出部分被丢弃 -> 趟3 跳过、趟4 兜底。
__global__ void tkr_gather_kernel(
        const float * __restrict__ src,
        const unsigned * __restrict__ prefix,   // T16, 低 16 位恒 0
        float * __restrict__ gathered_val, int * __restrict__ gathered_idx,
        int * __restrict__ gcount,
        const int ncols, const int bpr, const int k) {
    const int b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const unsigned T16 = prefix[row];
    const float * rs = src + (size_t)row * ncols;
    const int cs = b * P, ce = min(cs + P, ncols);
    float * gv = gathered_val + (size_t)row * CAPF;
    int   * gi = gathered_idx + (size_t)row * CAPF;

    int loc = 0;
    for (int idx = cs + t; idx < ce; idx += BLOCK)
        if ((f2k(rs[idx]) & 0xffff0000u) >= T16) loc++;   // 高 16 位 >= T16

    // warp inclusive scan -> 每线程 exclusive 写入偏移
    int val = loc;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, val, off);
        if ((t & 31) >= off) val += n;
    }
    __shared__ int s_warpsum[BLOCK/32];
    const int warp = t >> 5, lane = t & 31;
    if (lane == 31) s_warpsum[warp] = val;
    __syncthreads();
    int warp_start = 0;
    for (int w = 0; w < warp; w++) warp_start += s_warpsum[w];
    const int thread_start = warp_start + val - loc;

    // 块内归约 cnt(总命中)
    int c = loc;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
    __shared__ int s_red[BLOCK/32];
    if (lane == 0) s_red[warp] = c;
    __syncthreads();
    if (t < BLOCK/32) {
        int w = s_red[t];
        constexpr unsigned mask = 0xffu;
        #pragma unroll
        for (int off = BLOCK/32/2; off > 0; off >>= 1) w += __shfl_down_sync(mask, w, off);
        if (t == 0) s_red[0] = w;
    }
    __syncthreads();
    const int cnt = s_red[0];
    // base 必须广播到所有线程: 用 shared(只有 warp0/lane0 做 atomicAdd, 其他 warp 的 lane0 base=0)。
    __shared__ int s_base;
    if (t == 0) {
        s_base = atomicAdd(&gcount[row], cnt);
    }
    __syncthreads();
    const int base = s_base;

    int myrank = 0;
    for (int idx = cs + t; idx < ce; idx += BLOCK) {
        if ((f2k(rs[idx]) & 0xffff0000u) >= T16) {
            const int r = base + thread_start + myrank++;
            if (r < CAPF) { gv[r] = rs[idx]; gi[r] = idx; }
        }
    }
}

// 趟3: 每行 1 block, bitonic 排序 gathered(gcount 个, <=CAPF) 取前 k 写 dst。固定共享 CAPF*8。
// gcount[row] > CAPF 的行: gathered 已被截断, 本 kernel 不写 dst(交给 tkr_rowsel_kernel)。
__global__ void tkr_final_kernel(
        const float * __restrict__ gathered_val, const int * __restrict__ gathered_idx,
        const int * __restrict__ gcount,
        int * __restrict__ dst,
        const int nrows, const int k) {
    const int row = blockIdx.x, t = threadIdx.x;
    const int n = gcount[row];
    if (n > CAPF) return;
    const float * gv = gathered_val + (size_t)row * CAPF;
    const int   * gi = gathered_idx + (size_t)row * CAPF;

    __shared__ float s_val[CAPF];
    __shared__ int   s_idx[CAPF];

    int npad = 1;
    while (npad < n && npad < CAPF) npad <<= 1;
    for (int i = t; i < n; i += BLOCK) { s_val[i]=gv[i]; s_idx[i]=gi[i]; }
    for (int i = n + t; i < npad; i += BLOCK) { s_val[i]=-INFINITY; s_idx[i]=-1; }
    __syncthreads();
    bitonic_sort_pairs(s_val, s_idx, npad, t);
    int * out = dst + (size_t)row * k;
    for (int r = t; r < k; r += BLOCK) out[r] = s_idx[npad - 1 - r];
}

// 趟4: overflow 行(超大 tie)兜底, 全程 device 内决策。每行 1 block。
// 对 64-bit 复合键 key64 = ((u64)f2k(v) << 32) | (u32)idx 做 8 趟 8bit radix-select, 求第 k 大 key64 阈值 T;
// 再扫一遍 gather 所有 key64 >= T 的索引 —— 恰好 k 个(idx 唯一 -> key64 唯一 -> 无 tie 歧义), 顺序任意。
// dst 顺序不受契约约束(test_top_k 明确按集合比较), 故无需再排序。
// 仅在 overflow 行执行, 属罕见路径(真实连续 LID 分数 gathered≈700~1044 < CAPF), 用朴素 8 趟扫描换确定性。
__global__ void tkr_rowsel_kernel(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ gcount,
        const int ncols, const int k) {
    const int row = blockIdx.x, t = threadIdx.x;
    if (gcount[row] <= CAPF) return;                 // 常规行已由 tkr_final_kernel 写定

    const float * rs = src + (size_t)row * ncols;
    __shared__ int s_hist[256];
    __shared__ unsigned long long s_pre;
    __shared__ int s_rem;
    __shared__ int s_cnt;

    if (t == 0) { s_pre = 0ull; s_rem = k; s_cnt = 0; }
    __syncthreads();

    for (int pass = 0; pass < 8; pass++) {
        const int pos = 56 - pass * 8;
        const int hibits = pos + 8;                  // 已经定位的高位个数(0..64)
        for (int i = t; i < 256; i += BLOCK) s_hist[i] = 0;
        __syncthreads();
        const unsigned long long pre = s_pre;
        for (int idx = t; idx < ncols; idx += BLOCK) {
            const unsigned long long key = ((unsigned long long) f2k(rs[idx]) << 32) | (unsigned) idx;
            if (hibits >= 64 || (key >> hibits) == (pre >> hibits)) {   // 与已定高位一致
                atomicAdd(&s_hist[(int) ((key >> pos) & 0xffull)], 1);
            }
        }
        __syncthreads();
        if (t == 0) {
            int cum = 0, group = 0, r = s_rem;
            for (int d = 255; d >= 0; d--) {
                cum += s_hist[d];
                if (cum >= r) { group = d; r -= (cum - s_hist[d]); break; }
            }
            s_pre = pre | ((unsigned long long) group << pos);
            s_rem = r;
        }
        __syncthreads();
    }

    const unsigned long long T = s_pre;
    for (int idx = t; idx < ncols; idx += BLOCK) {
        const unsigned long long key = ((unsigned long long) f2k(rs[idx]) << 32) | (unsigned) idx;
        if (key >= T) {
            const int p = atomicAdd(&s_cnt, 1);
            if (p < k) dst[(size_t) row * k + p] = idx;
        }
    }
}

// init: 每行 prefix=0, rem=k(找第 k 大, 0-indexed 大值在前)。
__global__ void tkr_init_kernel(unsigned * prefix, int * rem, const int nrows, const int k) {
    const int row = blockIdx.x;
    prefix[row] = 0;
    rem[row] = k;
}

inline size_t tkr_lhist_bytes(int nrows, int bpr) {
    return (size_t)nrows * bpr * 256 * sizeof(int);
}
inline size_t tkr_gathered_bytes(int nrows) {
    return (size_t)nrows * CAPF * (sizeof(float) + sizeof(int));
}

// host launcher. 工作区全部取自 ggml_cuda_pool。
// 图安全: 只发射 kernel/Async memset, 无 host 读回、无同步 -> 可在 CUDA Graph capture 区内调用。
// overflow(某行 gathered>CAPF)由 tkr_rowsel_kernel 在 device 上兜底, host 无需知道。
inline void tkr_launch(ggml_cuda_pool & pool,
        const float * src, int * dst,
        int ncols, int nrows, int k, cudaStream_t stream) {
    const int bpr = (ncols + P - 1) / P;

    ggml_cuda_pool_alloc<int>      l_hist_alloc(pool, tkr_lhist_bytes(nrows, bpr));
    ggml_cuda_pool_alloc<unsigned> prefix_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int>      rem_alloc(pool, nrows);
    ggml_cuda_pool_alloc<float>    gval_alloc(pool, tkr_gathered_bytes(nrows));
    ggml_cuda_pool_alloc<int>      gidx_alloc(pool, tkr_gathered_bytes(nrows));
    ggml_cuda_pool_alloc<int>      gcount_alloc(pool, nrows);

    int *      l_hist = l_hist_alloc.get();
    unsigned * prefix = prefix_alloc.get();
    int *      rem    = rem_alloc.get();
    float *    gval   = gval_alloc.get();
    int *      gidx   = gidx_alloc.get();
    int *      gcount = gcount_alloc.get();

    // 趟1: 2 趟 8bit digit(MSB 两字节, pos=24,16), 每趟 hist + digit 两 kernel。读 4->2。
    tkr_init_kernel<<<nrows, 1, 0, stream>>>(prefix, rem, nrows, k);
    const int pos_list[2] = {24, 16};
    for (int i = 0; i < 2; i++) {
        tkr_hist_kernel<<<dim3(bpr, nrows), BLOCK, 0, stream>>>(
            src, l_hist, prefix, ncols, bpr, pos_list[i]);
        tkr_digit_kernel<<<nrows, BLOCK, 0, stream>>>(
            l_hist, prefix, rem, nrows, bpr, pos_list[i]);
    }

    // 趟2: gather(高16位>=T16)。
    cudaMemsetAsync(gcount, 0, (size_t)nrows * sizeof(int), stream);
    tkr_gather_kernel<<<dim3(bpr, nrows), BLOCK, 0, stream>>>(
        src, prefix, gval, gidx, gcount, ncols, bpr, k);

    // 趟3: final(非 overflow 行)。趟4: overflow 行 device 内兜底(非 overflow 行立即返回)。
    tkr_final_kernel<<<nrows, BLOCK, 0, stream>>>(
        gval, gidx, gcount, dst, nrows, k);
    tkr_rowsel_kernel<<<nrows, BLOCK, 0, stream>>>(
        src, dst, gcount, ncols, k);
}

} // namespace ggml_lid_topk_fast

#endif // GGML_CUDA_LID_TOP_K_FAST_H
