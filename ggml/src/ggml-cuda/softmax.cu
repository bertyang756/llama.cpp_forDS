#include "common.cuh"
#include "ggml.h"
#include "softmax.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif // GGML_USE_HIP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid  = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    //TODO: noncontigous inputs/outputs
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = sinks ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = block_reduce<block_reduce_method::MAX, block_size_template>(max_val, buf_iw);

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    if (block_size > WARP_SIZE) {
        // sync is needed as we reuse buf_iw across block_reduce invocations, see #26385
        // for block_size <= WARP_SIZE, block_reduce does not access buf_iw
        __syncthreads();
    }
    // find the sum of exps in the block
    tmp = block_reduce<block_reduce_method::SUM, block_size_template>(tmp, buf_iw);

    if (sinks) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}

// TODO: Template to allow keeping ncols in registers if they fit
static __device__ void soft_max_f32_parallelize_cols_single_row(const float * __restrict__ x,
                                                                float * __restrict__ dst,
                                                                float * __restrict__ tmp_maxs,
                                                                float * __restrict__ tmp_sums,
                                                                float * shared_vals_max,
                                                                float * shared_vals_sum,
                                                                const soft_max_params p) {
    namespace cg = cooperative_groups;

    const cg::grid_group g = cg::this_grid();

    const int tid               = threadIdx.x;
    const int col_start         = blockIdx.x * blockDim.x + tid;
    const int n_elem_per_thread = 4;

    float     local_vals[n_elem_per_thread] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float     local_max                     = -INFINITY;
    const int step_size                     = gridDim.x * blockDim.x;

    // Compute thread-local max
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            local_max = fmaxf(local_max, local_vals[i]);
        }
        col += step_size * n_elem_per_thread;
    }

    // Compute CTA-level max
    local_max = block_reduce<block_reduce_method::MAX>(local_max, shared_vals_max);

    // Store CTA-level max to GMEM
    if (tid == 0) {
        tmp_maxs[blockIdx.x] = local_max;
    }
    g.sync();

    // Compute compute global max from CTA-level maxs
    assert(gridDim.x < blockDim.x);  // currently we only support this case
    if (tid < gridDim.x) {
        local_max = tmp_maxs[tid];
    } else {
        local_max = -INFINITY;
    }
    local_max = block_reduce<block_reduce_method::MAX>(local_max, shared_vals_max);

    // Compute softmax dividends, accumulate divisor
    float tmp_expf = 0.0f;
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.ncols) {
                const float tmp = expf(local_vals[i] - local_max);
                tmp_expf += tmp;
                dst[idx] = tmp;
            }
        }
        col += step_size * n_elem_per_thread;
    }

    // Reduce divisor within CTA
    tmp_expf = block_reduce<block_reduce_method::SUM>(tmp_expf, shared_vals_sum);

    // Store CTA-level sum to GMEM
    if (tid == 0) {
        tmp_sums[blockIdx.x] = tmp_expf;
    }
    g.sync();

    // Compute global sum from CTA-level sums
    if (tid < gridDim.x) {
        tmp_expf = tmp_sums[tid];
    } else {
        tmp_expf = 0.0f;
    }
    tmp_expf = block_reduce<block_reduce_method::SUM>(tmp_expf, shared_vals_sum);

    // Divide dividend by global sum + store data
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? dst[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.ncols) {
                dst[idx] = local_vals[i] / tmp_expf;
            }
        }
        col += step_size * n_elem_per_thread;
    }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static __global__ void soft_max_back_f32(
        const float * grad, const float * dstf, float * dst, const int ncols, const float scale) {
    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template<int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, 0, 0, T>), smpbo);
    soft_max_f32<true, 0, 0><<<block_nums, block_dims, nbytes_shared, stream>>>(x, mask, sinks, dst, p);
}

__launch_bounds__(8*WARP_SIZE, 1) static __global__ void soft_max_f32_parallelize_cols(const float * __restrict__ x,
                                                     float * __restrict__ dst,
                                                     float * __restrict__ tmp_maxs,
                                                     float * __restrict__ tmp_sums,
                                                     const soft_max_params p)
// We loop over all instead of parallelizing across gridDim.y as cooperative groups
// currently only support synchronizing the complete grid if not launched as a cluster group
// (which requires CC > 9.0)
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#grid-synchronization
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#class-cluster-group
{
    __shared__ float shared_vals[2][32];

    for (int rowx = 0; rowx < p.ne01 * p.ne02 * p.ne03; rowx++) {
        soft_max_f32_parallelize_cols_single_row(x + int64_t(rowx) * p.ncols, dst + int64_t(rowx) * p.ncols, tmp_maxs,
                                                 tmp_sums, shared_vals[0], shared_vals[1], p);
    }
}

// ============================================================================
// Optimized fast paths for the plain, contiguous, f32 case:
//   scale == 1, no mask, no sinks, no ALiBi, row stride == ncols.
// These are tried before the generic soft_max_f32 kernels:
//   - one warp per row, whole row in registers                 (64 .. 1024)
//   - one warp handles 16 rows, 2 lanes/row, ncols == 8        (NARROW8)
// The build uses -use_fast_math, so expf() lowers to the SFU intrinsic.
// ============================================================================
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra-semi"
#endif // defined(__GNUC__)

namespace {

__device__ __forceinline__ float soft_max_fast_max4(const float4 v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

// --- narrow rows: one warp per row, values kept in registers, vectorised ---
template <int ITEMS, int WARPS>
__global__ void soft_max_fast_warp_reg(const float * __restrict__ x, float * __restrict__ y,
                                       int nrows, int ncols) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int wid  = threadIdx.x / WARP_SIZE;
    const int64_t row = (int64_t) blockIdx.x * WARPS + wid;
    if (row >= nrows) {
        return;
    }

    const float * xr = x + row * (int64_t) ncols;
    float *       yr = y + row * (int64_t) ncols;

    float v[ITEMS];
    if constexpr (ITEMS % 4 == 0) {
        const float4 * x4 = reinterpret_cast<const float4 *>(xr);
#pragma unroll
        for (int i = 0; i < ITEMS / 4; ++i) {
            const float4 t = x4[lane + i * WARP_SIZE];
            v[4 * i + 0] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
        }
    } else if constexpr (ITEMS % 2 == 0) {
        const float2 * x2 = reinterpret_cast<const float2 *>(xr);
#pragma unroll
        for (int i = 0; i < ITEMS / 2; ++i) {
            const float2 t = x2[lane + i * WARP_SIZE];
            v[2 * i + 0] = t.x; v[2 * i + 1] = t.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            v[i] = xr[lane + i * WARP_SIZE];
        }
    }

    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        max_val = fmaxf(max_val, v[i]);
    }
    max_val = warp_reduce_max(max_val);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        v[i] = expf(v[i] - max_val);
        sum += v[i];
    }
    sum = warp_reduce_sum(sum);
    const float inv_sum = 1.0f / sum;

    if constexpr (ITEMS % 4 == 0) {
        float4 * y4 = reinterpret_cast<float4 *>(yr);
#pragma unroll
        for (int i = 0; i < ITEMS / 4; ++i) {
            float4 o;
            o.x = v[4 * i + 0] * inv_sum; o.y = v[4 * i + 1] * inv_sum;
            o.z = v[4 * i + 2] * inv_sum; o.w = v[4 * i + 3] * inv_sum;
            y4[lane + i * WARP_SIZE] = o;
        }
    } else if constexpr (ITEMS % 2 == 0) {
        float2 * y2 = reinterpret_cast<float2 *>(yr);
#pragma unroll
        for (int i = 0; i < ITEMS / 2; ++i) {
            float2 o;
            o.x = v[2 * i + 0] * inv_sum;
            o.y = v[2 * i + 1] * inv_sum;
            y2[lane + i * WARP_SIZE] = o;
        }
    } else {
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            yr[lane + i * WARP_SIZE] = v[i] * inv_sum;
        }
    }
}

// --- very narrow rows (ncols == 8): one warp handles 16 rows, 2 lanes per
//     row, each lane loads one float4.  1 read + 1 write, no shared memory,
//     no block sync.  Targets the real-model majority shape.
template <int WARPS>
__global__ void soft_max_fast_narrow8(const float * __restrict__ x, float * __restrict__ y, int nrows) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x >> 5;
    const int gwar = blockIdx.x * WARPS + warp;
    const int row  = gwar * 16 + (lane >> 1);   // 16 rows per warp
    const int half = lane & 1;                   // 0: first float4, 1: second float4

    // keep all lanes active (avoid __shfl_sync full-mask UB); invalid rows
    // participate with neutral values and skip the store.
    const bool valid = row < nrows;
    float4 v;
    if (valid) {
        v = *reinterpret_cast<const float4 *>(x + (size_t) row * 8 + half * 4);
    } else {
        v = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    float m = valid ? soft_max_fast_max4(v) : -INFINITY;
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));   // pair reduce -> row max

    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
    float s = 0.f;
    if (valid) {
        e0 = expf(v.x - m); e1 = expf(v.y - m);
        e2 = expf(v.z - m); e3 = expf(v.w - m);
        s  = e0 + e1 + e2 + e3;
    }
    s += __shfl_xor_sync(0xffffffff, s, 1);            // pair reduce -> row sum

    const float inv = 1.0f / s;
    if (valid) {
        float4 o;
        o.x = e0 * inv; o.y = e1 * inv; o.z = e2 * inv; o.w = e3 * inv;
        *reinterpret_cast<float4 *>(y + (size_t) row * 8 + half * 4) = o;
    }
}


static int soft_max_fast_max_threads_per_sm() {
    static int cached_dev = -1;
    static int cached_val = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev != cached_dev) {
        cudaDeviceGetAttribute(&cached_val, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        cached_dev = dev;
    }
    return cached_val;
}

// =====================================================================
// Fast-path decision logging (GGML_CUDA_FAST_LOG).
// Records, at every fast-path threshold decision, the full execution
// conditions plus the chosen kernel / fallback reason, so a real model
// run can confirm whether the fast path is used and what conditions
// actually need optimization.
//   GGML_CUDA_FAST_LOG=1        -> emit to stderr, dedup by condition
//   GGML_CUDA_FAST_LOG=<path>   -> append to file, dedup by condition
//   value containing "full"     -> emit every occurrence (no dedup)
// Default off; when off there is only a single boolean check per call.
// =====================================================================
enum soft_max_fast_decision {
    SOFT_MAX_FAST_WARP    = 2,
    SOFT_MAX_FAST_NARROW8 = 12,
    SOFT_MAX_GEN_SHARED   = 6,
    SOFT_MAX_GEN_FALLBACK = 7,
    SOFT_MAX_COOP         = 8,
    SOFT_MAX_DISABLED     = 9,
    SOFT_MAX_NO_MATCH     = 10,
    SOFT_MAX_BACKWARD     = 11,
};

static const char * soft_max_fast_decision_name(int d) {
    switch (d) {
        case SOFT_MAX_FAST_WARP:    return "WARP";
        case SOFT_MAX_FAST_NARROW8: return "NARROW8";
        case SOFT_MAX_GEN_SHARED:   return "GEN_SHARED";
        case SOFT_MAX_GEN_FALLBACK: return "GEN_FALLBACK";
        case SOFT_MAX_COOP:         return "COOP";
        case SOFT_MAX_DISABLED:     return "DISABLED";
        case SOFT_MAX_NO_MATCH:     return "NO_MATCH";
        case SOFT_MAX_BACKWARD:     return "BACKWARD";
        default:                    return "?";
    }
}

#define SOFT_MAX_FAST_LOG_MAX 1024

static bool   soft_max_fast_log_enabled = false;
static bool   soft_max_fast_log_full    = false;
static FILE * soft_max_fast_log_fp      = nullptr;

static void soft_max_fast_log_dump_summary();

static void soft_max_fast_log_init() {
    static std::once_flag once;
    std::call_once(once, []() {
        const char * v = getenv("GGML_CUDA_FAST_LOG");
        if (v == nullptr) {
            return;
        }
        soft_max_fast_log_enabled = true;
        soft_max_fast_log_full    = (strstr(v, "full") != nullptr);
        if (strcmp(v, "1") == 0 || strcmp(v, "full") == 0) {
            soft_max_fast_log_fp = stderr;
        } else {
            soft_max_fast_log_fp = fopen(v, "a");
            if (soft_max_fast_log_fp == nullptr) {
                soft_max_fast_log_fp = stderr;
            }
        }
        std::atexit(soft_max_fast_log_dump_summary);
    });
}

static std::mutex soft_max_fast_log_mtx;

struct soft_max_fast_log_key {
    int64_t ncols;
    int64_t nrows;
    uint8_t contiguous;
    uint8_t mask_type;         // 0 none, 1 f32, 2 f16
    uint8_t has_sinks;
    uint8_t coop_would_apply;
    float   scale;
    float   max_bias;
    uint8_t decision;
};

static soft_max_fast_log_key soft_max_fast_log_keys[SOFT_MAX_FAST_LOG_MAX] = {};
static uint64_t              soft_max_fast_log_counts[SOFT_MAX_FAST_LOG_MAX] = {};
static int                   soft_max_fast_log_n = 0;

// --- aggregate histograms for shape-discovery / decision coverage ---
// Keyed by ncols (forward) and by decision id.  These are always accumulated
// (even in dedup mode) so a post-run summary can answer:
//   - which ncols actually appear and how often
//   - how many calls actually went through a designed fast path vs fallback
static int64_t  soft_max_fast_log_ncols[SOFT_MAX_FAST_LOG_MAX] = {};
static uint64_t soft_max_fast_log_ncols_count[SOFT_MAX_FAST_LOG_MAX] = {};
static int      soft_max_fast_log_ncols_n = 0;

static uint64_t soft_max_fast_log_decision_count[SOFT_MAX_FAST_LOG_MAX] = {};
static uint64_t soft_max_fast_log_total_calls = 0;

static void soft_max_fast_log_dump_summary() {
    if (!soft_max_fast_log_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lk(soft_max_fast_log_mtx);
    FILE * fp = soft_max_fast_log_fp ? soft_max_fast_log_fp : stderr;
    fprintf(fp, "[fastlog] SUMMARY total_calls=%llu\n",
            (unsigned long long) soft_max_fast_log_total_calls);

    // per-ncols forward histogram
    for (int i = 0; i < soft_max_fast_log_ncols_n; ++i) {
        fprintf(fp, "[fastlog] SUM ncols=%lld calls=%llu\n",
                (long long) soft_max_fast_log_ncols[i],
                (unsigned long long) soft_max_fast_log_ncols_count[i]);
    }

    // per-decision histogram (fast vs fallback coverage)
    for (int d = 0; d <= SOFT_MAX_FAST_NARROW8; ++d) {
        const uint64_t c = soft_max_fast_log_decision_count[d];
        if (c == 0) {
            continue;
        }
        fprintf(fp, "[fastlog] SUM decision=%s calls=%llu\n",
                soft_max_fast_decision_name(d), (unsigned long long) c);
    }
}

static void soft_max_fast_log_decision(int64_t nrows, int64_t ncols, bool contiguous, int mask_type,
                                       bool has_sinks, float scale, float max_bias,
                                       bool coop_would_apply, int decision, const char * reason) {
    if (!soft_max_fast_log_enabled) {
        return;
    }

    soft_max_fast_log_key k;
    k.ncols            = ncols;
    k.nrows            = nrows;
    k.contiguous       = contiguous ? 1 : 0;
    k.mask_type        = (uint8_t) mask_type;
    k.has_sinks        = has_sinks ? 1 : 0;
    k.coop_would_apply = coop_would_apply ? 1 : 0;
    k.scale            = scale;
    k.max_bias         = max_bias;
    k.decision         = (uint8_t) decision;

    std::lock_guard<std::mutex> lk(soft_max_fast_log_mtx);

    // aggregate histograms (always, even in dedup mode)
    soft_max_fast_log_total_calls++;
    if (decision >= 0 && decision <= SOFT_MAX_FAST_NARROW8) {
        soft_max_fast_log_decision_count[decision]++;
    }
    {
        int ni = -1;
        for (int i = 0; i < soft_max_fast_log_ncols_n; ++i) {
            if (soft_max_fast_log_ncols[i] == ncols) {
                ni = i;
                break;
            }
        }
        if (ni < 0) {
            if (soft_max_fast_log_ncols_n < SOFT_MAX_FAST_LOG_MAX) {
                ni = soft_max_fast_log_ncols_n++;
                soft_max_fast_log_ncols[ni] = ncols;
                soft_max_fast_log_ncols_count[ni] = 0;
            }
        }
        if (ni >= 0) {
            soft_max_fast_log_ncols_count[ni]++;
        }
    }

    int idx = -1;
    for (int i = 0; i < soft_max_fast_log_n; ++i) {
        if (memcmp(&soft_max_fast_log_keys[i], &k, sizeof(k)) == 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        if (soft_max_fast_log_n >= SOFT_MAX_FAST_LOG_MAX) {
            return;
        }
        idx = soft_max_fast_log_n++;
        soft_max_fast_log_keys[idx]     = k;
        soft_max_fast_log_counts[idx]   = 0;
    }
    soft_max_fast_log_counts[idx]++;

    if (soft_max_fast_log_full || soft_max_fast_log_counts[idx] == 1) {
        FILE * fp = soft_max_fast_log_fp ? soft_max_fast_log_fp : stderr;
        fprintf(fp, "[fastlog] DEC ncols=%lld nrows=%lld contig=%d mask=%d sinks=%d scale=%.4f maxbias=%.4f coop=%d => %s%s%s count=%llu\n",
                (long long) ncols, (long long) nrows,
                (int) k.contiguous, (int) k.mask_type, (int) k.has_sinks,
                k.scale, k.max_bias, (int) k.coop_would_apply,
                soft_max_fast_decision_name(decision),
                (reason && reason[0]) ? " " : "",
                reason ? reason : "",
                (unsigned long long) soft_max_fast_log_counts[idx]);
    }
}

// Backward has no fast path; only record per-ncols call counts.
static void soft_max_fast_log_backward(int ncols, int nrows) {
    if (!soft_max_fast_log_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lk(soft_max_fast_log_mtx);
    static int      b_ncols[SOFT_MAX_FAST_LOG_MAX] = {};
    static uint64_t b_counts[SOFT_MAX_FAST_LOG_MAX] = {};
    static int      b_n = 0;

    int idx = -1;
    for (int i = 0; i < b_n; ++i) {
        if (b_ncols[i] == ncols) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        if (b_n >= SOFT_MAX_FAST_LOG_MAX) {
            return;
        }
        idx = b_n++;
        b_ncols[idx]   = ncols;
        b_counts[idx]  = 0;
    }
    b_counts[idx]++;

    if (soft_max_fast_log_full || b_counts[idx] == 1) {
        FILE * fp = soft_max_fast_log_fp ? soft_max_fast_log_fp : stderr;
        fprintf(fp, "[fastlog] BACKWARD ncols=%d nrows=%d count=%llu\n", ncols, nrows,
                (unsigned long long) b_counts[idx]);
    }
}

// Runtime WARPS selection for the ncols=8 fast kernel.
// WARPS=8 (256 threads/block) is the default; CMP 170HX (2048 threads/SM)
// uses WARPS=16 (512 threads/block) for large row counts to improve occupancy.
// SOFTMAX_NARROW8_WARPS=<n> overrides for experiments.
static void launch_soft_max_fast_narrow8(const float * x, float * y, int nrows, int warps, cudaStream_t stream) {
    const int rows_per_block = warps * 16;
    const dim3 grid((nrows + rows_per_block - 1) / rows_per_block);
    switch (warps) {
        case 8:
            soft_max_fast_narrow8<8><<<grid, 8 * WARP_SIZE, 0, stream>>>(x, y, nrows);
            break;
        case 16:
            soft_max_fast_narrow8<16><<<grid, 16 * WARP_SIZE, 0, stream>>>(x, y, nrows);
            break;
        default:
            soft_max_fast_narrow8<8><<<grid, 8 * WARP_SIZE, 0, stream>>>(x, y, nrows);
            break;
    }
}

static bool soft_max_try_fast(const float * x, float * y, int nrows, int ncols, cudaStream_t stream,
                              int * out_decision) {
    if (!ggml_cuda_fast_ops_enabled("GGML_CUDA_FAST_SOFTMAX") || nrows <= 0 || ncols <= 0) {
        *out_decision = SOFT_MAX_DISABLED;
        return false;
    }

    // ncols == 8: very narrow rows, one warp handles 16 rows, 2 lanes/row
    if (ncols == 8) {
        int warps = 8;
        const char * env = getenv("SOFTMAX_NARROW8_WARPS");
        if (env) {
            warps = atoi(env);
        } else if (nrows > 128 && soft_max_fast_max_threads_per_sm() >= 2048) {
            warps = 16;   // CMP 170HX: 2048 threads/SM -> more warps/block
        }
        *out_decision = SOFT_MAX_FAST_NARROW8;
        launch_soft_max_fast_narrow8(x, y, nrows, warps, stream);
        return true;
    }

    // narrow rows: one warp per row, row in registers
    switch (ncols) {
        case 64:   *out_decision = SOFT_MAX_FAST_WARP; soft_max_fast_warp_reg<2,  8><<<(nrows + 7) / 8, 256, 0, stream>>>(x, y, nrows, ncols); return true;
        case 128:  *out_decision = SOFT_MAX_FAST_WARP; soft_max_fast_warp_reg<4,  8><<<(nrows + 7) / 8, 256, 0, stream>>>(x, y, nrows, ncols); return true;
        case 256:  *out_decision = SOFT_MAX_FAST_WARP; soft_max_fast_warp_reg<8,  8><<<(nrows + 7) / 8, 256, 0, stream>>>(x, y, nrows, ncols); return true;
        case 512:  *out_decision = SOFT_MAX_FAST_WARP; soft_max_fast_warp_reg<16, 8><<<(nrows + 7) / 8, 256, 0, stream>>>(x, y, nrows, ncols); return true;
        case 1024: *out_decision = SOFT_MAX_FAST_WARP; soft_max_fast_warp_reg<32, 4><<<(nrows + 3) / 4, 128, 0, stream>>>(x, y, nrows, ncols); return true;
        default: break;
    }


    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;


    // Anything that still fits in shared memory is left to the generic
    // soft_max_f32 kernels (they keep the row in SMEM, 1 read + 1 write).
    // The online/hybrid/reg paths were removed because the target model logs
    // only use ncols == 8 and ncols == 128 (see softmax_op/README.md §9.6).
    if ((size_t) (GGML_PAD(ncols, WARP_SIZE) + WARP_SIZE) * sizeof(float) <= smpbo) {
        *out_decision = SOFT_MAX_GEN_SHARED;
        return false;
    }


    *out_decision = SOFT_MAX_NO_MATCH;
    return false;
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif // defined(__GNUC__)

} // namespace

template <typename T>
static void soft_max_f32_cuda(const float *                                x,
                              const T *                                    mask,
                              const float *                                sinks,
                              float *                                      dst,
                              const soft_max_params &                      params,
                              cudaStream_t                                 stream,
                              [[maybe_unused]] ggml_backend_cuda_context & ctx,
                              const bool                                   contiguous) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    // Optimized fast paths for the plain, contiguous, mask-free case.  They are
    // skipped when the cooperative row-parallel path below would be selected,
    // because that path targets very large ncols with very few rows.

    // ---- fast-path threshold decision + logging (GGML_CUDA_FAST_LOG) ----
    soft_max_fast_log_init();

    const bool fast_eligible =
        contiguous && mask == nullptr && sinks == nullptr &&
        params.scale == 1.0f && params.max_bias == 0.0f &&
        ncols_x > 0 && ncols_x <= INT32_MAX;

    const int64_t nrows_x = params.ne01 * params.ne02 * params.ne03;
    const bool coop_would_apply =
        ggml_cuda_info().devices[id].supports_cooperative_launch &&
        nrows_x > 0 && ncols_x / nrows_x > 8192;

    const int mask_type = (mask != nullptr) ? (std::is_same<T, half>::value ? 2 : 1) : 0;

    int  decision = SOFT_MAX_GEN_FALLBACK;
    const char * reason = nullptr;

    if (!fast_eligible) {
        if (!contiguous)                reason = "contiguous=false";
        else if (mask)                  reason = "mask";
        else if (sinks)                 reason = "sinks";
        else if (params.scale != 1.0f)  reason = "scale!=1";
        else if (params.max_bias != 0.0f) reason = "alibi";
        else                            reason = "invalid-ncols";
        decision = SOFT_MAX_GEN_FALLBACK;
    } else if (coop_would_apply) {
        decision = SOFT_MAX_COOP;
        reason   = "coop";
    } else if (nrows_x <= 0 || nrows_x > INT32_MAX) {
        decision = SOFT_MAX_GEN_FALLBACK;
        reason   = "nrows-invalid";
    } else {
        int d = SOFT_MAX_GEN_FALLBACK;
        const bool used = soft_max_try_fast(x, dst, (int) nrows_x, (int) ncols_x, stream, &d);
        decision = d;
        if (used) {
            soft_max_fast_log_decision(nrows_x, ncols_x, contiguous, mask_type,
                                       sinks != nullptr, params.scale, params.max_bias,
                                       coop_would_apply, decision, nullptr);
            return;
        }
        reason = (d == SOFT_MAX_GEN_SHARED) ? "SKIP-fits-smem" : "no-fast-path";
    }

    // generic soft_max_f32 kernels will handle this call; log the fallback decision
    soft_max_fast_log_decision(nrows_x, ncols_x, contiguous, mask_type,
                               sinks != nullptr, params.scale, params.max_bias,
                               coop_would_apply, decision, reason);

    if (nbytes_shared <= smpbo) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
    } else {
        // Parallelize across SMs for top-p/dist-sampling
        // The heuristic for parallelizing rows across SMs vs parallelizing single row & looping over all rows was done on the basis of a B6000 GPU and
        // Can be adapted further for lower-SM-count GPUs, though keeping data in registers should be implemented first as that is the optimal solution.
        if (ggml_cuda_info().devices[id].supports_cooperative_launch &&
            ncols_x / (params.ne01 * params.ne02 * params.ne03) > 8192 && mask == nullptr && sinks == nullptr &&
            params.scale == 1.0f && params.max_bias == 0.0f) {
            ggml_cuda_pool_alloc<float> tmp_maxs_alloc(ctx.pool(), ggml_cuda_info().devices[id].nsm * sizeof(float));
            ggml_cuda_pool_alloc<float> tmp_sums_alloc(ctx.pool(), ggml_cuda_info().devices[id].nsm * sizeof(float));

            void * kernel_args[] = { (void *) &x, (void *) &dst, (void *) &tmp_maxs_alloc.ptr,
                                     (void *) &tmp_sums_alloc.ptr, (void *) const_cast<soft_max_params *>(&params) };
            CUDA_CHECK(cudaLaunchCooperativeKernel((void *) soft_max_f32_parallelize_cols,
                                                   dim3(ggml_cuda_info().devices[id].nsm, 1, 1),
                                                   dim3(WARP_SIZE * 8, 1, 1), kernel_args, 0, stream));
        } else {
            const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);
            soft_max_f32<false, 0, 0>
                <<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, sinks, dst, params);
        }
    }
}

static void soft_max_back_f32_cuda(
        const float * grad, const float * dstf, float * dst,
        const int ncols, const int nrows, const float scale, cudaStream_t stream) {
    soft_max_fast_log_init();
    soft_max_fast_log_backward(ncols, nrows);

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows,     1, 1);

    soft_max_back_f32<<<block_nums, block_dims, 0, stream>>>(grad, dstf, dst, ncols, scale);
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    // The optimized fast paths require plain row-major f32 rows with stride ncols.
    const bool contiguous =
        src0->nb[0] == (size_t) sizeof(float) &&
        dst->nb[0]  == (size_t) sizeof(float) &&
        src0->nb[1] == (size_t) ne00 * sizeof(float) &&
        dst->nb[1]  == (size_t) ne00 * sizeof(float) &&
        dst->ne[0]  == ne00;

    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    if (use_f16) {
        soft_max_f32_cuda(src0_d, (const half *) src1_d, (const float *) src2_d, dst_d, params, stream, ctx, contiguous);
    } else {
        soft_max_f32_cuda(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream, ctx, contiguous);
    }
}

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_cuda(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
