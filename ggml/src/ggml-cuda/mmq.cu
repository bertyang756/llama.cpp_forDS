#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>


// Re-index the quantized activation (block_q8_1_mmq) rows from the global compact-row
// layout into a per-bucket local compact-row layout, so that the mmq kernel can keep its
// fast contiguous y loads.
//   src_y layout: [k_block][global_row][block]
//   dst_y layout: [k_block][local_row][block]
// row_ids maps local row -> global row.  The local buffer must be zeroed beforehand; only
// the first n_local_rows rows of each k-block are written.
static __global__ void mmq_reindex_y_kernel(
        const int * __restrict__ src_y, int * __restrict__ dst_y,
        const int32_t * __restrict__ row_ids,
        const int n_local_rows, const int n_global_rows, const int dst_stride_rows,
        const int n_kblocks, const int sz) {
    const int kb = blockIdx.x;
    const int r  = blockIdx.y;
    const int gr = row_ids[r];
    const int * src = src_y + ((int64_t) kb*n_global_rows + gr)*sz;
    int * dst = dst_y + ((int64_t) kb*dst_stride_rows + r)*sz;
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        dst[i] = src[i];
    }
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ncols_opt = ne11;
        if (ids) {
            const int64_t n_expert_used = ids->ne[0];
            ncols_opt = ne12;

            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
            // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
            if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
            }
        }

        int ntiles_J_best = INT_MAX;

        for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc, prec_src1);
            if (config.type == GGML_TYPE_COUNT) {
                continue;
            }

            if (mmq_get_nbytes_shared(config, cc) > smpbo) {
                continue;
            }

            const int ntiles_x = (ncols_opt + config.J - 1) / config.J;

            if (ntiles_x < ntiles_J_best) {
                J_best = J;
                nthreads_best = config.nthreads;
                ntiles_J_best = ntiles_x;
            }
        }
    }
    GGML_ASSERT(J_best > 0);

    // A tile of size J can read in at most J - 1 extra columns.
    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
    const size_t src1_load_chunk_size = nthreads_best * sizeof(int);
    const size_t src1_q8_1_padding = ((J_best * sizeof(block_q8_1_mmq) + src1_load_chunk_size - 1) / src1_load_chunk_size)
        * src1_load_chunk_size;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, J_best};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    if (ggml_cuda_moe_probe_type(src0->type)) {
        const int n_kblocks = (int) (ne10_padded / QK8_1_MMQ);
        GGML_LOG_INFO("moe-probe:   mmq: K=%d expert_rows=%d n_experts=%d ne10=%d ne11=%d T=%d n_expert_used=%d ne_get_rows=%d dedup_bcast=%d fallback=%d n_kblocks=%d\n",
                      (int) ne00, (int) ne01, (int) ne02, (int) ne10, (int) ne11, (int) ne12,
                      (int) n_expert_used, (int) ne_get_rows, (int) dedup_bcast, (int) fallback, n_kblocks);
    }

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // O2 fast path: per-expert J bucketing for the MoE ids branch with T>=64.
    // Split experts by col_diff (= how many active rows each expert gets) into a few
    // buckets, each launched with the smallest J >= bucket max col_diff.  This cuts the
    // J/col_diff mma waste that the global-J>=T launch pays for every expert.
    // y is re-laid out per bucket (gather) so the mmq kernel keeps its contiguous loads;
    // only the weight expert id needs indirection (expert_ids).
    //
    // Bucketing is computed on host from `ids` (a stable graph input, NOT a pool temp),
    // so it does NOT race with concurrent streams / CUDA graph capture the way reading
    // back the pool-temp expert_bounds/ids_dst did.  Still skip while capturing: the
    // host D2H of `ids` is not graph-capturable.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;

    const bool fast_bucket = ggml_cuda_fast_ops_enabled("GGML_CUDA_FAST_MOE_MM")
        && ggml_cuda_fast_ops_enabled("GGML_CUDA_FAST_MOEMM_O1O2")
        && ids != nullptr
        && ne12 >= MMQ_DP4A_MAX_BATCH_SIZE
        && !use_native_fp4
        && !is_capturing;

    if (ggml_cuda_moe_probe_type(src0->type)) {
        GGML_LOG_INFO("moe-probe:   fast_bucket=%d (FAST_MOE_MM=%d O1O2=%d ne12=%d>=%d ids=%d native_fp4=%d capturing=%d)\n",
                      (int) fast_bucket,
                      (int) ggml_cuda_fast_ops_enabled("GGML_CUDA_FAST_MOE_MM"),
                      (int) ggml_cuda_fast_ops_enabled("GGML_CUDA_FAST_MOEMM_O1O2"),
                      (int) ne12, MMQ_DP4A_MAX_BATCH_SIZE, (int) (ids != nullptr), (int) use_native_fp4, (int) is_capturing);
    }

    if (fast_bucket) {
        const int n_experts     = (int) ne02;
        const int n_expert_used = (int) ids->ne[0];
        const int n_tokens      = (int) ne12;
        const int n_rows_g      = n_tokens * n_expert_used;
        const int n_kblocks     = (int)(ne10_padded / QK8_1_MMQ);
        const int sz            = (int)(sizeof(block_q8_1_mmq) / sizeof(int));
        const int si1           = (int)(ids->nb[1] / ggml_element_size(ids));

        GGML_ASSERT(n_experts > 0 && n_experts < 1000000);
        GGML_ASSERT(n_rows_g  > 0 && n_rows_g  < 100000000);

        // read the routing ids (stable graph input) to host once
        std::vector<int32_t> ids_host((size_t) n_tokens * si1);
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, (size_t) n_tokens * si1 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // per-expert: count and dst columns (it*topk+iex), in token order (matches mm_ids_helper)
        std::vector<int> counts(n_experts, 0);
        std::vector<int> exp_base(n_experts, 0);
        std::vector<std::vector<int32_t>> dst_per_exp(n_experts);
        int compact = 0;
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int e = ids_host[(size_t) it*si1 + iex];
                GGML_ASSERT(e >= 0 && e < n_experts);
                counts[e]++;
                dst_per_exp[e].push_back(it*n_expert_used + iex);
                ++compact;
            }
        }
        GGML_ASSERT(compact == n_rows_g);
        int acc = 0;
        for (int e = 0; e < n_experts; ++e) { exp_base[e] = acc; acc += counts[e]; }

        // Bucket thresholds (J per bucket); experts with col_diff > 64 go to the last bucket
        // and switch_J will pick a larger J there.
        static const int thresholds[] = {16, 32, 48, 64};
        const int nbuckets = 4;
        std::vector<std::vector<int>> bucket_exp(nbuckets);
        int bucket_max[4] = {0, 0, 0, 0};
        for (int e = 0; e < n_experts; ++e) {
            if (counts[e] == 0) {
                continue;
            }
            int b = 0;
            while (b < nbuckets - 1 && counts[e] > thresholds[b]) {
                ++b;
            }
            bucket_exp[b].push_back(e);
            if (counts[e] > bucket_max[b]) {
                bucket_max[b] = counts[e];
            }
        }

        std::vector<int> nE(nbuckets, 0), nrows(nbuckets, 0);
        std::vector<std::vector<int32_t>> h_exp_ids(nbuckets), h_bounds_l(nbuckets), h_ids_dst_l(nbuckets), h_row_ids(nbuckets);
        for (int b = 0; b < nbuckets; ++b) {
            if (bucket_exp[b].empty()) {
                continue;
            }
            int lr = 0;
            for (int e : bucket_exp[b]) {
                lr += counts[e];
            }
            nE[b]    = (int) bucket_exp[b].size();
            nrows[b] = lr;
            h_exp_ids[b].resize(nE[b]);
            h_bounds_l[b].resize(nE[b] + 1);
            h_ids_dst_l[b].resize(lr);
            h_row_ids[b].resize(lr);
            int r = 0;
            h_bounds_l[b][0] = 0;
            for (int i = 0; i < nE[b]; ++i) {
                const int e = bucket_exp[b][i];
                h_exp_ids[b][i] = e;
                h_bounds_l[b][i] = r;
                for (int k = 0; k < counts[e]; ++k) {
                    h_ids_dst_l[b][r] = dst_per_exp[e][k];
                    h_row_ids[b][r]   = exp_base[e] + k; // global compact row (expert-sorted)
                    ++r;
                }
                h_bounds_l[b][i + 1] = r;
            }
        }

        // Padding rows per k-block so the mmq kernel can read J columns past the last row.
        const int pad = 128; // >= max J
        if (ggml_cuda_moe_probe_verbose() && ggml_cuda_moe_probe_type(src0->type)) {
            GGML_LOG_INFO("moe-probe:     buckets: thresholds={16,32,48,64} nbuckets=%d pad=%d\n", nbuckets, pad);
            for (int b = 0; b < nbuckets; ++b) {
                if (bucket_exp[b].empty()) continue;
                GGML_LOG_INFO("moe-probe:       bucket %d: nE=%d nrows=%d J=%d stride_rows=%d\n",
                              b, nE[b], nrows[b], bucket_max[b], nrows[b] + pad);
            }
        }
        for (int b = 0; b < nbuckets; ++b) {
            if (nE[b] == 0) {
                continue;
            }
            const int J          = bucket_max[b];
            const int stride_rows = nrows[b] + pad;
            const int nbuf_y      = n_kblocks * stride_rows * sz;

            ggml_cuda_pool_alloc<int>    d_y(ctx.pool(), nbuf_y);
            ggml_cuda_pool_alloc<int32_t> d_exp(ctx.pool(), nE[b]);
            ggml_cuda_pool_alloc<int32_t> d_bounds(ctx.pool(), nE[b] + 1);
            ggml_cuda_pool_alloc<int32_t> d_ids(ctx.pool(), nrows[b]);
            ggml_cuda_pool_alloc<int32_t> d_rows(ctx.pool(), nrows[b]);

            CUDA_CHECK(cudaMemsetAsync(d_y.get(), 0, (size_t) nbuf_y*sizeof(int), stream));
            CUDA_CHECK(cudaMemcpyAsync(d_exp.get(),   h_exp_ids[b].data(),   (size_t) nE[b]*sizeof(int32_t),        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_bounds.get(), h_bounds_l[b].data(),  (size_t) (nE[b] + 1)*sizeof(int32_t),  cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_ids.get(),   h_ids_dst_l[b].data(), (size_t) nrows[b]*sizeof(int32_t),     cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_rows.get(),  h_row_ids[b].data(),   (size_t) nrows[b]*sizeof(int32_t),     cudaMemcpyHostToDevice, stream));

            const dim3 gcopy((unsigned) n_kblocks, (unsigned) nrows[b]);
            mmq_reindex_y_kernel<<<gcopy, 256, 0, stream>>>(
                (const int *) src1_q8_1.get(), d_y.get(), d_rows.get(),
                nrows[b], n_rows_g, stride_rows, n_kblocks, sz);
            CUDA_CHECK(cudaGetLastError());

            mmq_args bk_args;
            bk_args.x = src0_d;
            bk_args.type_x = src0->type;
            bk_args.y = d_y.get();
            bk_args.ids_dst = d_ids.get();
            bk_args.expert_bounds = d_bounds.get();
            bk_args.expert_ids = d_exp.get();
            bk_args.dst = dst_d;
            bk_args.y_scale = nullptr;
            bk_args.ncols_x = ne00;
            bk_args.nrows_x = ne01;
            bk_args.ncols_dst = nrows[b];
            bk_args.stride_row_x = s01;
            bk_args.ncols_y = stride_rows;   // y row stride per k-block
            bk_args.nrows_dst = s1;          // dst column stride
            bk_args.nchannels_x = nE[b];
            bk_args.nchannels_y = nE[b];
            bk_args.stride_channel_x = s02;
            bk_args.stride_channel_y = 0;    // unused in the ids branch
            bk_args.stride_channel_dst = s2;
            bk_args.nsamples_x = ne03;
            bk_args.nsamples_y = ne13;
            bk_args.stride_sample_x = s03;
            bk_args.stride_sample_y = s13;
            bk_args.stride_sample_dst = s3;
            bk_args.ncols_max = J;           // max col_diff in bucket -> switch_J picks J>=J, ntx=1

            bk_args.J_best = J;              // per-bucket tile width (upstream moved J_best to args)
            ggml_cuda_mul_mat_q_switch_type(ctx, bk_args, stream, prec_src1);
        }
        return; // handled by the bucketed path
    }



    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), nullptr, dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, J_best};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

