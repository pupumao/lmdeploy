# Copyright (c) OpenMMLab. All rights reserved.
# include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../../3rdparty/trt_llm_lib/include/)
# include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../../3rdparty/trt_llm_lib/include/tensorrt_llm/cutlass_extensions/include/)
# link_directories(${CMAKE_CURRENT_LIST_DIR}/../../../../3rdparty/trt_llm_lib/lib/)

add_library(gemm_s4_f8 STATIC gemm_s4_f8.cc)
# target_compile_options(gemm_s4_f8 PRIVATE
#   --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
set_property(TARGET gemm_s4_f8 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm_s4_f8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(gemm_s4_f8 PUBLIC fpA_intB_gemm_src_lib)
