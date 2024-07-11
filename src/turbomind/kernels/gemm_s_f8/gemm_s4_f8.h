// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/macro.h"
#include "src/turbomind/utils/allocator.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace turbomind {

extern bool g_dump_kernel_info_once;

class GemmS4F8 {
public:
    GemmS4F8(IAllocator* allocator);

    ~GemmS4F8();

    void Run(half*        C,
             const uint*  A,
             const half*  B,
             const half*  Q,
             const half*  Z,
             const float  alpha,
             int          m,
             int          n,
             int          k,
             int          group_size,
             cudaStream_t st);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
