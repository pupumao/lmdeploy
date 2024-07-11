// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

namespace turbomind {

template<typename T>
struct res_norm_ops_t {
};

template<typename T>
struct res_norm_t {
    res_norm_ops_t<T> f;
    __device__ uint4  addvec(const uint4& a, const uint4& b, const uint4& bias, float& accum) const
    {
        uint4 c;
        c.x = f.cast(f.add(f.cast(a.x), f.cast(b.x), f.cast(bias.x), accum));
        c.y = f.cast(f.add(f.cast(a.y), f.cast(b.y), f.cast(bias.y), accum));
        c.z = f.cast(f.add(f.cast(a.z), f.cast(b.z), f.cast(bias.z), accum));
        c.w = f.cast(f.add(f.cast(a.w), f.cast(b.w), f.cast(bias.w), accum));
        return c;
    }
    __device__ uint4 normvec(const uint4& u, const uint4& s, float factor) const
    {
        uint4 v;
        v.x = f.cast(f.norm(f.cast(u.x), f.cast(s.x), factor));
        v.y = f.cast(f.norm(f.cast(u.y), f.cast(s.y), factor));
        v.z = f.cast(f.norm(f.cast(u.z), f.cast(s.z), factor));
        v.w = f.cast(f.norm(f.cast(u.w), f.cast(s.w), factor));
        return v;
    }
};

template<>
struct res_norm_ops_t<half> {
    __device__ float2 cast(const uint& x) const
    {
        return __half22float2(reinterpret_cast<const half2&>(x));
    }
    __device__ uint cast(const float2& x) const
    {
        auto y = __float22half2_rn(x);
        return reinterpret_cast<uint&>(y);
    }
    __device__ float2 add(const float2& a, const float2& b, const float2& bias, float& accum) const
    {
        float2 c{a.x + b.x + bias.x, a.y + b.y + bias.y};
        accum += c.x * c.x + c.y * c.y;
        return c;
    }
    __device__ float2 norm(const float2& a, const float2& s, float factor) const
    {
        return {a.x * s.x * factor, a.y * s.y * factor};
    }
};

template<>
struct res_norm_ops_t<float> {
    __device__ float cast(const uint& x) const
    {
        return reinterpret_cast<const float&>(x);
    }
    __device__ uint cast(const float& x) const
    {
        return reinterpret_cast<const uint&>(x);
    }
    __device__ float add(const float& a, const float& b, const float& bias, float& accum) const
    {
        float c = a + b + bias;
        accum += c * c;
        return c;
    }
    __device__ float norm(const float& a, const float& s, float factor) const
    {
        return a * s * factor;
    }
};

#ifdef ENABLE_BF16
template<>
struct res_norm_ops_t<__nv_bfloat16> {
    __device__ float2 cast(const uint& x) const
    {
        return cuda_cast<float2, __nv_bfloat162>(reinterpret_cast<const __nv_bfloat162&>(x));
    }
    __device__ uint cast(const float2& x) const
    {
        auto y = cuda_cast<__nv_bfloat162, float2>(x);
        return reinterpret_cast<uint&>(y);
    }
    __device__ float2 add(const float2& a, const float2& b, const float2& bias, float& accum) const
    {
        float2 c{a.x + b.x + bias.x, a.y + b.y + bias.y};
        accum += c.x * c.x + c.y * c.y;
        return c;
    }
    __device__ float2 norm(const float2& a, const float2& s, float factor) const
    {
        return {a.x * s.x * factor, a.y * s.y * factor};
    }
};

#endif

template<typename T>
__device__ T blockReduceSum(const cg::thread_block& block, T value)
{
    __shared__ float partial[32];

    auto tile = cg::tiled_partition<32>(block);
    value     = cg::reduce(tile, value, cg::plus<float>{});

    if (tile.thread_rank() == 0) {
        partial[tile.meta_group_rank()] = value;
    }

    block.sync();

    value = tile.thread_rank() < tile.meta_group_size() ? partial[tile.thread_rank()] : T{};
    return cg::reduce(tile, value, cg::plus<float>{});
}

// r' = r + x
// x' = norm(r') * scales
template<typename T>
__global__ void fusedAddBiasResidualNorm(T* __restrict__ r_data,
                                         T* __restrict__ x_data,
                                         const T* __restrict__ bias,
                                         const T* __restrict__ scale,
                                         float eps,
                                         int   batch_size,
                                         int   n_dims)
{
    auto block = cg::this_thread_block();
    auto grid  = cg::this_grid();

    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);

    const auto batch_idx            = block.group_index().x;
    uint4* __restrict__ r_ptr       = reinterpret_cast<uint4*>(r_data + batch_idx * n_dims);
    uint4* __restrict__ x_ptr       = reinterpret_cast<uint4*>(x_data + batch_idx * n_dims);
    const uint4* __restrict__ b_ptr = reinterpret_cast<const uint4*>(bias);

    res_norm_t<T> ops;

    float thread_sum{};
    for (auto i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
        auto  r  = r_ptr[i];
        auto  x  = x_ptr[i];
        uint4 b  = b_ptr ? b_ptr[i] : uint4{};
        r        = ops.addvec(r, x, b, thread_sum);
        r_ptr[i] = r;
    }

    auto total_sum = blockReduceSum(block, thread_sum);

    float s_inv_mean = rsqrt(total_sum / n_dims + eps);

    const uint4* __restrict__ s_ptr = reinterpret_cast<const uint4*>(scale);
    for (uint i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
        auto r   = r_ptr[i];
        auto s   = s_ptr[i];
        auto o   = ops.normvec(r, s, s_inv_mean);
        x_ptr[i] = o;
    }
}

template<typename T>
__global__ void fusedAddBiasResidualNorm(T* __restrict__ r_data,
                                         T* __restrict__ x_data,
                                         T* __restrict__ tmp_data,
                                         const T* __restrict__ bias,
                                         const T* __restrict__ scale,
                                         float eps,
                                         int   batch_size,
                                         int   n_dims) {
    auto block = cg::this_thread_block();
    auto grid  = cg::this_grid();
    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);
    const auto batch_idx            = grid.thread_rank() / block.size();
    uint4* __restrict__ r_ptr       = reinterpret_cast<uint4*>(r_data + batch_idx * n_dims);
    uint4* __restrict__ x_ptr       = reinterpret_cast<uint4*>(x_data + batch_idx * n_dims);
    const uint4* __restrict__ b_ptr = reinterpret_cast<const uint4*>(bias);
    uint4* __restrict__ x_ptr_fp16  = reinterpret_cast<uint4*>(tmp_data + batch_idx * n_dims);


    res_norm_t<T> ops; 
    float thread_sum{};
    for (auto i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
        auto  r  = r_ptr[i];
        auto  x  = x_ptr[i];
        uint4 b  = b_ptr ? b_ptr[i] : uint4{};
        r        = ops.addvec(r, x, b, thread_sum);
        r_ptr[i] = r;
    }
    auto total_sum = blockReduceSum(block, thread_sum);
    float s_inv_mean = rsqrt(total_sum / n_dims + eps);
    const uint4* __restrict__ s_ptr = reinterpret_cast<const uint4*>(scale);

    for (uint i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
        auto r   = r_ptr[i];
        auto s   = s_ptr[i];
        auto o   = ops.normvec(r, s, s_inv_mean);
        x_ptr_fp16[i] = o;
    }
}

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* in_out, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream)
{
    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);
    FT_CHECK(n_dims % PACK_DIM == 0);
    const int n_pack    = n_dims / PACK_DIM;
    const int n_iter    = ((n_pack + 1023) / 1024);        // iterations when block size == 1024
    int       n_threads = (n_pack + n_iter - 1) / n_iter;  // adjust block size to avoid tail effect
    n_threads           = (n_threads + 31) / 32 * 32;      // round up to the nearest multiple of warp size

    fusedAddBiasResidualNorm<<<batch_size, n_threads, 0, stream>>>(
        residual, in_out, bias, scale, eps, batch_size, n_dims);
}

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(T*           residual,
                                       T*           in_out,
                                       T*           quant_out,
                                       const T*     bias,
                                       const T*     scale,
                                       float        eps,
                                       int          batch_size,
                                       int          n_dims,
                                       cudaStream_t stream) {
    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);
    FT_CHECK(n_dims % PACK_DIM == 0);
    const int n_pack    = n_dims / PACK_DIM;
    const int n_iter    = ((n_pack + 1023) / 1024);        // iterations when block size == 1024
    int       n_threads = (n_pack + n_iter - 1) / n_iter;  // adjust block size to avoid tail effect
    n_threads           = (n_threads + 31) / 32 * 32;      // round up to the nearest multiple of warp size

    fusedAddBiasResidualNorm<<<batch_size, n_threads, 0, stream>>>(
        residual, in_out, quant_out, bias, scale, eps, batch_size, n_dims);
}

template<typename T>
__global__ void maskOutput(T* output, const int* mask, int dim)
{
    int batch_idx = blockIdx.x;
    output += dim * batch_idx;
    int masked = mask[batch_idx];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = (masked) ? output[i] : T();
    }
}

template<typename T>
void invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream)
{
    maskOutput<<<batch_size, 1024, 0, stream>>>(output, mask, dim);
}

template<typename T>
__global__ void generalT5LayerNormWithFP8PerChnQuantization(const T* __restrict input,
                                                            const T* __restrict gamma,
                                                            T*          output,
                                                            const float layernorm_eps,
                                                            int         m,
                                                            int         n,
                                                            T*      per_chn_scale) {
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(float)) char _shmem[];  // Align on largest type
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    // auto block = cg::this_thread_block();
    // variance = blockReduceSum(block, local_var_sum);
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    // V2
    // T                local_max = T(0.0f);
    // __shared__ float s_global_max;

    size_t token_id = blockIdx.x;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        // 加载激活值
        T val = clamp_inf_for_half<T>((((float)input[token_id * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
        // 激活值暂存到共享内存
        shmem[i] = val;
        // 计算局部的激活值的绝对值最大值
        // local_max = cuda_max(local_max, cuda_abs(val));
    }
    __syncthreads();

    // // 线程块级别归约局部最大值得到全局最大值
    // float global_max = blockReduceMax((float)local_max);
    // if (threadIdx.x == 0) {
    //     s_global_max = global_max;
    // }
    // __syncthreads();

    // float output_scale = 127.0f / s_global_max;
    // scale[blockIdx.x]  = 1.0f / output_scale;

    // __syncthreads();

    for (size_t i = tid; i < n; i += blockDim.x) {
        // TODO read per-chn-scale
        float output_scale                                           = (float)per_chn_scale[n + i];
        float val                                                    = (float)(shmem[i]) * output_scale;
        reinterpret_cast<__nv_fp8_e4m3*>(output)[blockIdx.x * n + i] = static_cast<__nv_fp8_e4m3>(val);
    }
}

template<typename T>
__global__ void generalAddResidualT5LayerNormWithFP8PerChnQuantization(const T* __restrict input,  // layer_input
                                                                       const T* __restrict gamma,
                                                                       T*          output,       // self_attn_output
                                                                       T*          norm_output,  // decoder_normed_input
                                                                       const float layernorm_eps,
                                                                       int         m,
                                                                       int         n,
                                                                       T*          per_chn_scale) {
    // layernorm module in the T5 style No bias and no subtraction of mean.
    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((float)ldg(&input[blockIdx.x * n + i]) + (float)output[blockIdx.x * n + i]);

        float diff = (float)(output[blockIdx.x * n + i]);
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    // 开始PerToken 计算 scale, 并用它量化norm_output, 最后传出scale用于GEMM
    extern __shared__ __align__(sizeof(float)) char _shmem[];  // Align on largest type
    T*                                              shmem = reinterpret_cast<T*>(_shmem);

    // __shared__ float s_global_max;
    // float            global_max = 0.0f;
    // float            local_max  = 0.0f;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        shmem[i]  = clamp_inf_for_half<T>((((float)output[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
        // local_max = cuda_max(local_max, cuda_abs((float)shmem[i]));
    }
    __syncthreads();
    // global_max = blockReduceMax(local_max);

    // if (threadIdx.x == 0) {
    //     s_global_max = global_max;
    // }
    // __syncthreads();

    // float output_scale = 127.0f / s_global_max;
    // scale[blockIdx.x]  = 1.0f / output_scale;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        // TODO read per-chn-scale
        float output_scale                                           = (float)per_chn_scale[n + i];
        float val                                                    = (float)(shmem[i]) * output_scale;
        reinterpret_cast<__nv_fp8_e4m3*>(output)[blockIdx.x * n + i] = static_cast<__nv_fp8_e4m3>(val);
    }
}

template<typename T>
void invokeGeneralLayerNormWithFP8PerChnQuantization(T*          out,
                                                     const T*    input,
                                                     const T*    gamma,
                                                     const T*    beta,
                                                     const float layernorm_eps,
                                                     const int   m,
                                                     const int   n,
                                                     T*          scale,
                                                     cudaStream_t stream)
{
    FT_CHECK(beta == nullptr);
    size_t shmem_size = n * sizeof(T);
    if (shmem_size >= (48 << 10)) {
        check_cuda_error(cudaFuncSetAttribute(generalT5LayerNormWithFP8PerChnQuantization<T>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              shmem_size));
    }

    dim3 grid(m);
    dim3 block(std::min(1024, n));
    generalT5LayerNormWithFP8PerChnQuantization<<<grid, block, shmem_size, stream>>>(
        input, gamma, out, layernorm_eps, m, n, scale);
}

template<typename T>
void invokeGeneralAddResidualLayerNormWithFP8PerChnQuantization(T*           output,       // 残差
                                                                T*           norm_output,  // 最终输出
                                                                const T*     input,        // 输入
                                                                const T*     gamma,
                                                                const float  layernorm_eps,
                                                                int          m,
                                                                int          n,
                                                                T*           scale,
                                                                cudaStream_t stream) {

    size_t shmem_size = n * sizeof(T);
    if (shmem_size >= (48 << 10)) {
        check_cuda_error(cudaFuncSetAttribute(generalAddResidualT5LayerNormWithFP8PerChnQuantization<T>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              shmem_size));
    }

    dim3 grid(m);
    dim3 block(min(n, 1024));

    generalAddResidualT5LayerNormWithFP8PerChnQuantization<T>
        <<<grid, block, shmem_size, stream>>>(input, gamma, output, norm_output, layernorm_eps, m, n, scale);
}

#ifdef ENABLE_FP32
template void
invokeFusedAddBiasResidualRMSNorm(float*, float*, const float*, const float*, float, int, int, cudaStream_t);
template void
invokeFusedAddBiasResidualRMSNorm(float*, float*, float*, const float*, const float*, float, int, int, cudaStream_t);

template void invokeMask(float* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
template void invokeGeneralLayerNormWithFP8PerChnQuantization(float*       out,
                                                              const float* input,
                                                              const float* gamma,
                                                              const float* beta,
                                                              const float  layernorm_eps,
                                                              const int    m,
                                                              const int    n,
                                                              float*       scale,
                                                              cudaStream_t stream);

template void invokeGeneralAddResidualLayerNormWithFP8PerChnQuantization(float*       output,
                                                                         float*       norm_output,
                                                                         const float* input,
                                                                         const float* gamma,
                                                                         const float  layernorm_eps,
                                                                         int          m,
                                                                         int          n,
                                                                         float*       scale,
                                                                         cudaStream_t stream);

#endif
template void invokeFusedAddBiasResidualRMSNorm(half*, half*, const half*, const half*, float, int, int, cudaStream_t);
template void invokeFusedAddBiasResidualRMSNorm(half*, half*, half*, const half*, const half*, float, int, int, cudaStream_t);

template void invokeMask(half* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
template void invokeGeneralLayerNormWithFP8PerChnQuantization(half*        out,
                                                              const half*  input,
                                                              const half*  gamma,
                                                              const half*  beta,
                                                              const float  layernorm_eps,
                                                              const int    m,
                                                              const int    n,
                                                              half*        scale,
                                                              cudaStream_t stream);

template void invokeGeneralAddResidualLayerNormWithFP8PerChnQuantization(half*        output,
                                                                         half*        norm_output,
                                                                         const half*  input,
                                                                         const half*  gamma,
                                                                         const float  layernorm_eps,
                                                                         int          m,
                                                                         int          n,
                                                                         half*        scale,
                                                                         cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeFusedAddBiasResidualRMSNorm(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);
template void invokeFusedAddBiasResidualRMSNorm(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);

template void invokeMask(__nv_bfloat16* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
template void invokeGeneralLayerNormWithFP8PerChnQuantization(__nv_bfloat16*       out,
                                                              const __nv_bfloat16* input,
                                                              const __nv_bfloat16* gamma,
                                                              const __nv_bfloat16* beta,
                                                              const float          layernorm_eps,
                                                              const int            m,
                                                              const int            n,
                                                              __nv_bfloat16*       scale,
                                                              cudaStream_t         stream);

template void invokeGeneralAddResidualLayerNormWithFP8PerChnQuantization(__nv_bfloat16*       output,
                                                                         __nv_bfloat16*       norm_output,
                                                                         const __nv_bfloat16* input,
                                                                         const __nv_bfloat16* gamma,
                                                                         const float          layernorm_eps,
                                                                         int                  m,
                                                                         int                  n,
                                                                         __nv_bfloat16*       scale,
                                                                         cudaStream_t         stream);
#endif
}  // namespace turbomind
