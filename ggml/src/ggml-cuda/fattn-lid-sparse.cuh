#pragma once
// O3 融合出席索引内核 —— v3：位图字扫描（word-wise bitmap scan）。
// 2026-10-05: 由沙盒 lid_topk_op/fattn-lid-sparse_v3.cuh 替换 v2（全扫+位图）。
// 正确性: 沙盒 verify_lid_v3 对 stock/v1/v2 逐位一致(7 组压力用例全 0 失配)。
//
// 背景（为什么 v2 还是慢）：
//   v2（当前集成版）虽然用 O(k) 建了 top_k 位图，但随后仍然**逐元素**扫完整个
//   n_concat = n_kv_raw + n_csa 区间。也就是说索引生产者保持 O(n_kv_raw + n_csa)，
//   和 stock 的 compact_mask 同量级，O3 唯一的收益只剩"删掉了掩码链"。
//   真实模型量级：n_kv_raw=768, n_csa≈3584(14K)~30000+(120K), k=512 —— 逐元素扫 n_csa
//   完全是浪费，因为候选集合本身就来自只有 k 项的 top_k 列表。
//
// v3 的做法：位图里"置位"就意味着 `index ∈ top_k && isfinite(csa[index])`（建位图时
//   顺手做掉 AND 过滤），于是 csa 段的输出就是位图的**升序展开**。按 **32 位字**扫描
//   （nw = ceil(n_csa/32) 次 popcount）而不是逐元素扫描，主导项直接砍 ~32×：
//     代价 = O(n_kv_raw) + O(nw + k)          (v2: O(n_kv_raw + n_csa))
//
// 输出与 v2 / stock **逐位一致**：raw 段在前（升序），csa 段在后（按位图字序 = 位置升序），
//   截断语义同 stock（写满 n_kv_max 即止，count = min(total, n_kv_max)）。
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace fattn_lid_sparse {

constexpr int LID_SPARSE_BLOCK = 256;
constexpr int LID_SPARSE_VPL   = 8;   // 仅 raw 段使用, 与 stock compact_mask 一致

// 每 (group, sequence) 一个 block（本路径要求 ncols1==1 ⇒ 每 block 一行）。
// 动态共享: 位图(nw u32) + warp 偏移(nwarp i32) + total/row_count。
static __global__ void lid_sparse_index_kernel(
        const int    * __restrict__ top_k,      // [k, n_groups, ?, n_seq]
        const half   * __restrict__ raw_mask,   // [n_kv_raw, n_groups, 1, n_seq]
        const half   * __restrict__ csa_mask,   // [n_csa,    n_groups, 1, n_seq]
        int          * __restrict__ out_idx,    // [n_seq*n_groups, n_kv_max]
        int          * __restrict__ out_cnt,    // [n_seq*n_groups]
        const int n_groups, const int k, const int n_csa, const int n_kv_raw, const int n_kv_max,
        const int64_t tk_s1, const int64_t tk_s3,
        const int64_t r_s1,  const int64_t r_s3,
        const int64_t c_s1,  const int64_t c_s3) {
    const int group = blockIdx.x;
    const int seq   = blockIdx.y;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int nt    = blockDim.x;
    const int nwarp = nt / 32;
    const int nw    = (n_csa + 31) / 32;

    extern __shared__ char smem_raw[];
    unsigned * s_bit   = (unsigned *) smem_raw;                        // top_k ∩ finite(csa) 位图
    int      * s_wo    = (int *) (smem_raw + (size_t) nw * 4);         // 每 warp 计数/偏移
    int      * s_total = s_wo + nwarp;
    int      * s_row   = s_total + 1;

    const int  * tk   = top_k + seq*tk_s3 + group*tk_s1;
    const half * raw  = raw_mask + seq*r_s3 + group*r_s1;
    const half * csa  = csa_mask + seq*c_s3 + group*c_s1;
    int        * idx  = out_idx + (int64_t)(seq*n_groups + group) * n_kv_max;

    // 1. 位图：只放"在 top_k 且 csa 有限"的位置（越界忽略、重复自动去重 ⇒ 与 v2 同解）
    for (int i = tid; i < nw; i += nt) s_bit[i] = 0;
    __syncthreads();
    for (int i = tid; i < k; i += nt) {
        const int p = tk[i];
        if (p >= 0 && p < n_csa && isfinite(__half2float(csa[p]))) {
            atomicOr(&s_bit[p >> 5], 1u << (p & 31));
        }
    }
    if (tid == 0) s_row[0] = 0;
    __syncthreads();

    // 2. raw 段：位置序 compact（与 v2 同构，区间缩到 n_kv_raw）
    for (int i0 = 0; i0 < n_kv_raw; i0 += nt * LID_SPARSE_VPL) {
        uint32_t sel[LID_SPARSE_VPL];
        int wc = 0;
#pragma unroll
        for (int item = 0; item < LID_SPARSE_VPL; ++item) {
            const int i = i0 + (warp*LID_SPARSE_VPL + item)*32 + lane;
            const bool s = i < n_kv_raw && isfinite(__half2float(raw[i]));
            sel[item] = __ballot_sync(0xFFFFFFFFu, s);
            wc += __popc(sel[item]);
        }
        if (lane == 0) s_wo[warp] = wc;
        __syncthreads();
        if (tid == 0) {
            int o = 0;
            for (int iw = 0; iw < nwarp; ++iw) { const int c = s_wo[iw]; s_wo[iw] = o; o += c; }
            s_total[0] = o;
        }
        __syncthreads();
        const uint32_t lm = lane == 0 ? 0u : (1u << lane) - 1u;
        int wi = 0;
#pragma unroll
        for (int item = 0; item < LID_SPARSE_VPL; ++item) {
            const int i = i0 + (warp*LID_SPARSE_VPL + item)*32 + lane;
            const int dst = s_row[0] + s_wo[warp] + wi + __popc(sel[item] & lm);
            if ((sel[item] & (uint32_t(1) << lane)) && dst < n_kv_max) idx[dst] = i;
            wi += __popc(sel[item]);
        }
        __syncthreads();
        if (tid == 0) s_row[0] += s_total[0];
        __syncthreads();
    }

    // 3. csa 段：按位图字序扫描（= 位置升序），每字权重 = popcount
    for (int w0 = 0; w0 < nw; w0 += nt) {
        const int w  = w0 + tid;
        const int pc = (w < nw) ? __popc(s_bit[w]) : 0;
        int scan = pc;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int n = __shfl_up_sync(0xFFFFFFFFu, scan, d);
            if (lane >= d) scan += n;
        }
        const int warp_total = __shfl_sync(0xFFFFFFFFu, scan, 31);  // lane 31 = warp 内 inclusive 总数
        if (lane == 0) s_wo[warp] = warp_total;
        __syncthreads();
        if (tid == 0) {
            int o = 0;
            for (int iw = 0; iw < nwarp; ++iw) { const int c = s_wo[iw]; s_wo[iw] = o; o += c; }
            s_total[0] = o;
        }
        __syncthreads();
        if (w < nw) {
            int dst = s_row[0] + s_wo[warp] + (scan - pc);
            unsigned bits = s_bit[w];
            while (bits) {
                const int b = __ffs(bits) - 1;
                bits &= bits - 1;
                if (dst < n_kv_max) idx[dst] = n_kv_raw + (w << 5) + b;
                ++dst;
            }
        }
        __syncthreads();
        if (tid == 0) s_row[0] += s_total[0];
        __syncthreads();
    }

    // 4. count + 越界 -1 填充
    const int count = min(s_row[0], n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += nt) idx[i] = -1;
    if (tid == 0) out_cnt[seq*n_groups + group] = count;
}

inline size_t lid_sparse_index_smem_bytes(int n_csa) {
    return ((size_t)((n_csa + 31) / 32) * 4) + (LID_SPARSE_BLOCK / 32) * 4 + 8;
}

inline void lid_sparse_index_launch(const void * top_k, const void * raw_mask, const void * csa_mask,
                           int * out_idx, int * out_cnt,
                           int n_groups, int n_seq, int k, int n_kv_raw, int n_csa, int n_kv_max,
                           int64_t tk_s1, int64_t tk_s3,
                           int64_t r_s1, int64_t r_s3, int64_t c_s1, int64_t c_s3,
                           cudaStream_t st) {
    const size_t smem = lid_sparse_index_smem_bytes(n_csa);
    if (smem + 1024 > 48 * 1024) {
        cudaFuncSetAttribute(lid_sparse_index_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(smem + 1024));
    }
    const dim3 grid(n_groups, n_seq, 1);
    lid_sparse_index_kernel<<<grid, LID_SPARSE_BLOCK, smem, st>>>(
        (const int *) top_k, (const half *) raw_mask, (const half *) csa_mask, out_idx, out_cnt,
        n_groups, k, n_csa, n_kv_raw, n_kv_max,
        tk_s1, tk_s3, r_s1, r_s3, c_s1, c_s3);
}

// ============================================================================
// 宽(union) 融合出席索引内核 —— prefill union 路线 (§9.2a A)。
// 2026-10-07: 由沙盒 lid_topk_op/union_lid_sparse.cuh 集成。
// 正确性: 沙盒 verify_union 对 compact_mask<8> union(CPU+GPU) 逐位一致(全 0 失配)。
// 语义: 每组 N=8 个 query 求并集可见列 → 1 条 union 索引(concat 空间)。
//   raw 段: 位置 i∈[0,n_kv_raw) 可见 iff ∃q∈组 isfinite(raw_q[i])。
//   csa 段: 位置 p∈[0,n_csa) 可见 iff ∃q∈组 (p ∈ top_k_q ∧ isfinite(csa_q[p]))。
//   (csa 段 = build_top_k_mask 的 add 语义: -inf 支配 ⇒ 并集掩码 finite ⟺ top_k∩因果)
// 输出 = raw 段位置序(0..n_kv_raw-1) 在前, csa 段按位图字序展开(n_kv_raw+位置) 在后,
//   写满 n_kv_max 截断, count=min(total,n_kv_max), 越界 -1 填充 —— 与 compact_mask<8> 逐位一致。
// 代价: O(N×(n_kv_raw+k) + N×nw) (nw=ceil(n_csa/32)) —— 免逐元素扫全 n_csa(8×n_csa)。
// 数据布局(镜像 fattn-common 调用): top_k [k,n_queries,?,n_seq]; raw_mask = src[3] concat 并集
//   掩码(r_s1=n_kv_raw+n_csa); csa_mask = src[5] CSA 因果掩码(c_s1=n_csa); 每组 N=8 query。
constexpr int LID_UNION_BLOCK = 256;
constexpr int LID_UNION_N     = 8;   // queries per group (宽 sparse ncols1)

// 每 (group, sequence) 一个 block。
// 动态共享: csa 并集位图(nw u32) + raw 并集位图(nr u32) + warp 偏移(nwarp i32) + total/row_count。
static __global__ void union_lid_sparse_index_kernel(
        const int    * __restrict__ top_k,      // [k, n_queries, ?, n_seq]
        const half   * __restrict__ raw_mask,   // [n_kv_raw+n_csa, n_queries, 1, n_seq] concat 掩码
        const half   * __restrict__ csa_mask,   // [n_csa, n_queries, 1, n_seq]
        int          * __restrict__ out_idx,    // [n_seq*n_groups, n_kv_max]
        int          * __restrict__ out_cnt,    // [n_seq*n_groups]
        const int n_groups, const int k, const int n_csa, const int n_kv_raw, const int n_kv_max,
        const int64_t tk_s1, const int64_t tk_s3,
        const int64_t r_s1,  const int64_t r_s3,
        const int64_t c_s1,  const int64_t c_s3) {
    const int group = blockIdx.x;
    const int seq   = blockIdx.y;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int nt    = blockDim.x;
    const int nwarp = nt / 32;
    const int nw    = (n_csa + 31) / 32;
    const int nr    = (n_kv_raw + 31) / 32;
    const int q0    = group * LID_UNION_N;

    extern __shared__ char smem_raw[];
    unsigned * s_csa   = (unsigned *) smem_raw;                        // 并集 csa 位图
    unsigned * s_raw   = (unsigned *) (smem_raw + (size_t) nw * 4);    // 并集 raw 位图
    int      * s_wo    = (int *) (smem_raw + (size_t)(nw + nr) * 4);   // 每 warp 计数/偏移
    int      * s_total = s_wo + nwarp;
    int      * s_row   = s_total + 1;

    // 1. 并集位图: csa 段 = OR_{q∈组} (top_k_q ∩ finite(csa_q)); raw 段 = OR_{q∈组} finite(raw_q)
    for (int i = tid; i < nw; i += nt) s_csa[i] = 0;
    for (int i = tid; i < nr; i += nt) s_raw[i] = 0;
    __syncthreads();
    for (int idx = tid; idx < LID_UNION_N * k; idx += nt) {
        const int q = idx / k, i = idx % k;
        const int p = top_k[seq*tk_s3 + (int64_t)(q0 + q)*tk_s1 + i];
        if (p >= 0 && p < n_csa && isfinite(__half2float(csa_mask[seq*c_s3 + (int64_t)(q0 + q)*c_s1 + p]))) {
            atomicOr(&s_csa[p >> 5], 1u << (p & 31));
        }
    }
    for (int idx = tid; idx < LID_UNION_N * n_kv_raw; idx += nt) {
        const int q = idx / n_kv_raw, i = idx % n_kv_raw;
        if (isfinite(__half2float(raw_mask[seq*r_s3 + (int64_t)(q0 + q)*r_s1 + i]))) {
            atomicOr(&s_raw[i >> 5], 1u << (i & 31));
        }
    }
    if (tid == 0) s_row[0] = 0;
    __syncthreads();

    // 2. raw 段: 按位图字序扫描(=位置升序), 输出 0..n_kv_raw-1
    for (int w0 = 0; w0 < nr; w0 += nt) {
        const int w  = w0 + tid;
        const int pc = (w < nr) ? __popc(s_raw[w]) : 0;
        int scan = pc;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int n = __shfl_up_sync(0xFFFFFFFFu, scan, d);
            if (lane >= d) scan += n;
        }
        const int warp_total = __shfl_sync(0xFFFFFFFFu, scan, 31);
        if (lane == 0) s_wo[warp] = warp_total;
        __syncthreads();
        if (tid == 0) {
            int o = 0;
            for (int iw = 0; iw < nwarp; ++iw) { const int c = s_wo[iw]; s_wo[iw] = o; o += c; }
            s_total[0] = o;
        }
        __syncthreads();
        if (w < nr) {
            int dst = s_row[0] + s_wo[warp] + (scan - pc);
            unsigned bits = s_raw[w];
            while (bits) {
                const int b = __ffs(bits) - 1;
                bits &= bits - 1;
                if (dst < n_kv_max) out_idx[(int64_t)(seq*n_groups + group) * n_kv_max + dst] = b + (w << 5);
                ++dst;
            }
        }
        __syncthreads();
        if (tid == 0) s_row[0] += s_total[0];
        __syncthreads();
    }

    // 3. csa 段: 按位图字序扫描(=位置升序), 输出 n_kv_raw + 位置
    for (int w0 = 0; w0 < nw; w0 += nt) {
        const int w  = w0 + tid;
        const int pc = (w < nw) ? __popc(s_csa[w]) : 0;
        int scan = pc;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int n = __shfl_up_sync(0xFFFFFFFFu, scan, d);
            if (lane >= d) scan += n;
        }
        const int warp_total = __shfl_sync(0xFFFFFFFFu, scan, 31);
        if (lane == 0) s_wo[warp] = warp_total;
        __syncthreads();
        if (tid == 0) {
            int o = 0;
            for (int iw = 0; iw < nwarp; ++iw) { const int c = s_wo[iw]; s_wo[iw] = o; o += c; }
            s_total[0] = o;
        }
        __syncthreads();
        if (w < nw) {
            int dst = s_row[0] + s_wo[warp] + (scan - pc);
            unsigned bits = s_csa[w];
            while (bits) {
                const int b = __ffs(bits) - 1;
                bits &= bits - 1;
                if (dst < n_kv_max) out_idx[(int64_t)(seq*n_groups + group) * n_kv_max + dst] = n_kv_raw + (w << 5) + b;
                ++dst;
            }
        }
        __syncthreads();
        if (tid == 0) s_row[0] += s_total[0];
        __syncthreads();
    }

    // 4. count + 越界 -1 填充
    const int count = min(s_row[0], n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += nt)
        out_idx[(int64_t)(seq*n_groups + group) * n_kv_max + i] = -1;
    if (tid == 0) out_cnt[seq*n_groups + group] = count;
}

inline size_t union_lid_sparse_smem_bytes(int n_csa, int n_kv_raw) {
    return ((size_t)((n_csa + 31) / 32) * 4) + ((size_t)((n_kv_raw + 31) / 32) * 4) + (LID_UNION_BLOCK / 32) * 4 + 8;
}

inline void union_lid_sparse_index_launch(const void * top_k, const void * raw_mask, const void * csa_mask,
                           int * out_idx, int * out_cnt,
                           int n_groups, int n_seq, int k, int n_kv_raw, int n_csa, int n_kv_max,
                           int64_t tk_s1, int64_t tk_s3,
                           int64_t r_s1, int64_t r_s3, int64_t c_s1, int64_t c_s3,
                           cudaStream_t st) {
    const size_t smem = union_lid_sparse_smem_bytes(n_csa, n_kv_raw);
    if (smem + 1024 > 48 * 1024) {
        cudaFuncSetAttribute(union_lid_sparse_index_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(smem + 1024));
    }
    const dim3 grid(n_groups, n_seq, 1);
    union_lid_sparse_index_kernel<<<grid, LID_UNION_BLOCK, smem, st>>>(
        (const int *) top_k, (const half *) raw_mask, (const half *) csa_mask, out_idx, out_cnt,
        n_groups, k, n_csa, n_kv_raw, n_kv_max,
        tk_s1, tk_s3, r_s1, r_s3, c_s1, c_s3);
}

} // namespace fattn_lid_sparse
