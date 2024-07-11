#include "gemm_s4_f8.h"
#include "cutlass/array.h"
#include "cutlass/util/device_memory.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

namespace turbomind{
using FpAIntBGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using FpAIntBGemmRunnerPtr = std::shared_ptr<FpAIntBGemmRunner>;
namespace tkc = tensorrt_llm::cutlass_extensions;

struct GemmS4F8::Impl {
    Impl(IAllocator* allocator): allocator_(allocator) {
        gemm_s4_fp8_asym_ = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
            __nv_fp8_e4m3,
            cutlass::uint4b_t,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS,
            half,
            half,
            half>>();
        gemm_s4_fp8_sym_  = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
             __nv_fp8_e4m3,
             cutlass::uint4b_t,
             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
             half,
             half,
             half>>();
    }
    ~Impl(){
        if (fpaintb_workspace_) {
            allocator_->free((void**)&fpaintb_workspace_);
            fpaintb_workspace_ = nullptr;
            fpaintb_workspace_size_ = -1;
        }
    }

    void Run(half*        C,
             const uint*  A,
             const half*  B,
             const half*    Q,
             const half*    Z,
             const float alpha,
             int          m,
             int          n,
             int          k,
             int          group_size,
             cudaStream_t st){
        auto& gemm    = *gemm_s4_fp8_sym_;
        // auto  configs = getConfigs(gemm, weight.input_dims);
        auto  configs = getConfigs(gemm, k);
        // TODO some config choose logic
        auto  choose_config = configs[0];
        // int const ws_bytes      = gemm_s4_fp8_sym_->getWorkspaceSize(n, weight.output_dims, weight.input_dims);
        int const ws_bytes      = gemm_s4_fp8_sym_->getWorkspaceSize(n, m, k);
        // int       alloc_ws_bytes =
        //     ws_bytes + static_cast<size_t>(batch_size) * static_cast<size_t>(weight.input_dims) * sizeof(float);
        int alloc_ws_bytes = ws_bytes + static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float);
        if (!fpaintb_workspace_ || alloc_ws_bytes != fpaintb_workspace_size_) {
            fpaintb_workspace_ = (char*)allocator_->reMalloc(fpaintb_workspace_, alloc_ws_bytes);
            fpaintb_workspace_size_ = alloc_ws_bytes;
        }
        gemm_s4_fp8_sym_->gemm(B,           // input activation ptr
                               A,           // int4 weight ptr
                               Q,           // weight scales ptr
                               Z,           // weight zeros ptr
                               nullptr,     // bias ptr
                               alpha,       // alpha, use to scale output
                               C,           // output activation ptr
                               n,           // num tokens(m)
                               m,           // out dims(n)
                               k,           // in dims(k)
                               group_size,  // group size
                               choose_config,
                               fpaintb_workspace_,
                               fpaintb_workspace_size_,
                               st);
    }

private:
    template<typename T>
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> getConfigs(T& runner, int k){
        auto                                                             configs = runner.getConfigs();
        std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> rets;
        for (auto config : configs) {
            // std::cout << "config to filter, current: " << config.toString() << std::endl;
            if (config.stages == 2)
                continue;
            if (config.split_k_style != tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K) {
                int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
                if (k_size % 64)
                    continue;
            }
            rets.push_back(config);
        }
        return rets;
    }
private:
    IAllocator*    allocator_;

    FpAIntBGemmRunnerPtr gemm_s4_fp8_sym_;
    FpAIntBGemmRunnerPtr gemm_s4_fp8_asym_;

    char* fpaintb_workspace_ = nullptr;
    int   fpaintb_workspace_size_ = -1;
};

GemmS4F8::GemmS4F8(IAllocator* allocator): impl_(std::make_unique<Impl>(allocator)){}

GemmS4F8::~GemmS4F8(){}

void GemmS4F8::Run(half*        C,
                   const uint*  A,
                   const half*  B,
                   const half*  Q,
                   const half*  Z,
                   const float  alpha,
                   int          m,
                   int          n,
                   int          k,
                   int          group_size,
                   cudaStream_t st){
    impl_->Run(C, A, B, Q, Z, alpha, m, n, k, group_size, st);
}
}