// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void invokeGeneralLayerNormWithFP8PerChnQuantization(T*           out,
                                                     const T*     input,
                                                     const T*     gamma,
                                                     const T*     beta,
                                                     const float  layernorm_eps,
                                                     const int    m,
                                                     const int    n,
                                                     T*           scale,
                                                     cudaStream_t stream);


template<typename T>
void invokeGeneralAddResidualLayerNormWithFP8PerChnQuantization(T*           output,       // 残差
                                                                T*           norm_output,  // 最终输出
                                                                const T*     input,        // 输入
                                                                const T*     gamma,
                                                                const float  layernorm_eps,
                                                                int          m,
                                                                int          n,
                                                                T*           scale,
                                                                cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* in_out, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* in_out, T* quant_out, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream);

template<typename T>
void invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream);

}  // namespace turbomind
