
#include "cta_gemm.cuh"
#include "hz_fused_decl.cuh"
#include "tile_scheduler.cuh"

namespace mxmoe {
using namespace tiled_gemm;
using MatricesInfo=RCR_FP16FP16FP16;


// template<TileConfig<128,128,64,4,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
__global__ void groupgemm_hz_fused_4_impl (     //
    half** ptr_As,                       //
    half** ptr_Bs,                       //
    half** ptr_scale_zp_a,               //
    half** ptr_scale_zp_b,               //
    MatricesInfo::TC** ptr_Cs,           //
    MatricesInfo::TC** ptr_Ds,           //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    dim3* problem_sizes,                 //
    const QParams* qbits_list,           //
    int problem_count,                   //
    
    int* problem_tiles_prefix_sum        //
) 
        {
        using TA = typename MatricesInfo::TA;
        using TB = typename MatricesInfo::TB;
        using TC = typename MatricesInfo::TC;
        static_assert(std::is_same_v<TA, TB> && std::is_same_v<TA, half>, "");
        constexpr auto LAYOUT_A = MatricesInfo::LAYOUT_A;
        constexpr auto LAYOUT_B = MatricesInfo::LAYOUT_B;
        constexpr auto LAYOUT_C = MatricesInfo::LAYOUT_C;

        int lane = threadIdx.x;
        int tidx = lane + threadIdx.y * WarpSize;

        extern __shared__ uint8_t smem[];
        auto smem_scale_zp = reinterpret_cast<half*>(smem + 98304);

        auto visitor  = TileScheduler(problem_sizes, problem_count, problem_tiles_prefix_sum);
        auto tile_idx = blockIdx.x;
        auto cta_size = gridDim.x;

        while (true) {
            // get corresponding [tileA, tileB, tileC] offset
            auto problem_idx = visitor.get_problem_idx(tile_idx);
            // early exit if all problems are done
            if (problem_idx == -1) break;

            // [act, w]
            int a_bits = qbits_list[problem_idx].qbits.x;
            int w_bits = qbits_list[problem_idx].qbits.y;
            int gsize  = qbits_list[problem_idx].gsize;
            bool sym   = qbits_list[problem_idx].sym;

            
            // get coordinate of TileC of current problem
            auto tile_coord = [&] {
              if (a_bits == 16 && w_bits == 16) return visitor.get_tile_coord<128, 128>(problem_idx, tile_idx);

              return dim3(0, 0, 0);
            }();
        
            
            auto problem_size = problem_sizes[problem_idx];
            auto M            = problem_size.x;
            auto N            = problem_size.y;
            auto K            = problem_size.z;

            if (a_bits == 16 && w_bits == 16) {
              using TileCfg = TileConfig<128,128,64,4,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>;
              using TileA = GemmTileA<TA, LAYOUT_A, TileCfg>;
              using TileB = GemmTileB<TB, LAYOUT_B, TileCfg>;
              using TileC = GlobalTileC<TC, LAYOUT_C, TileCfg>;

              auto warp_x = TileCfg::warp_x();
              auto warp_y = TileCfg::warp_y();
              auto warp_z = TileCfg::warp_z();

              auto tile_a = TileA(ptr_As[problem_idx], M, K, tile_coord.y, tile_coord.x, warp_y, warp_z, tidx);
              auto tile_b = TileB(ptr_Bs[problem_idx], N, K, tile_coord.y, tile_coord.x, warp_x, warp_z, tidx);
              auto tile_c = TileC(ptr_Cs[problem_idx], M, N, tile_coord.y, tile_coord.x, tidx);

              

              cta_gemm_multistage_v2<TileCfg>(tile_a, tile_b, tile_c, smem, K, lane, warp_x, warp_y, warp_z);
            }
            

            // advance
            tile_idx += cta_size;
        }
        }
        




// template<>
void groupgemm_hz_fused_4(                      //
    MatricesInfo::TA** ptr_As,           //
    MatricesInfo::TB** ptr_Bs,           //
    half** ptr_scale_zp_a,               //
    half** ptr_scale_zp_b,               //
    MatricesInfo::TC** ptr_Cs,           //
    MatricesInfo::TC** ptr_Ds,           //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    dim3* problem_sizes,                 //
    dim3* h_problem_sizes,               //
    QParams* qbits_list,                 //
    QParams* h_qbits_list,               //
    int problem_count                    //
) {
  using TA = MatricesInfo::TA;
  using TB = MatricesInfo::TB;
  using TC = MatricesInfo::TC;

  auto problem_tiles_prefix_sum = new int[problem_count];
  auto total_tiles              = 0;
  for (int i = 0; i < problem_count; i++) {
    auto a_bits = h_qbits_list[i].qbits.x;
    auto w_bits = h_qbits_list[i].qbits.y;
    auto sym    = h_qbits_list[i].sym;
    auto gsize  = h_qbits_list[i].gsize;

    int M = h_problem_sizes[i].x;
    int N = h_problem_sizes[i].y;

    if (a_bits == 16 && w_bits == 16) total_tiles+=cu_cdiv(M,128)*cu_cdiv(N,128);else throw std::runtime_error("quant type not supported");

    problem_tiles_prefix_sum[i] = total_tiles;
  }
  int* d_problem_tiles_prefix_sum;
  checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * problem_count));
  checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum, problem_count * sizeof(int), cudaMemcpyHostToDevice));

  // auto kernel     = groupgemm_hz_fused_4_impl<TileConfig<128,128,64,4,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>;
  auto kernel     = groupgemm_hz_fused_4_impl;

  int dev_id;
  int num_sm;
  int max_active_blocks;
  int num_threads=32*8;
  size_t smem_size=98304;
  checkCudaErrors(cudaGetDevice(&dev_id));
  checkCudaErrors(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, num_threads, smem_size));
  int num_ctas = num_sm;
  // int num_ctas = num_sm * num_ctas_per_sm;
  auto launch_cfg = cudaLaunchConfig_t{
      .gridDim          = dim3(num_ctas),
      .blockDim         = dim3(32, 8),
      .dynamicSmemBytes = smem_size,
      .stream           = 0,  // use default stream
      .attrs            = nullptr,
      .numAttrs         = 0,  // no attributes
  };
  checkCudaErrors(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  checkCudaErrors(cudaLaunchKernelEx(          //
      &launch_cfg, kernel,                     //
      ptr_As, ptr_Bs,                          //
      ptr_scale_zp_a,                          //
      ptr_scale_zp_b,                          //
      ptr_Cs, ptr_Ds,                          //
      ldas, ldbs, ldcs, ldds,                  //
      problem_sizes,                           //
      qbits_list,                              //
      problem_count,                           //
      d_problem_tiles_prefix_sum  //
      ));

  delete[] problem_tiles_prefix_sum;
  checkCudaErrors(cudaFree(d_problem_tiles_prefix_sum));
}


/////////////////////////////////////////////////////


struct Register_groupgemm_hz_fused_4{Register_groupgemm_hz_fused_4(){register_kernel<TileConfig<128,128,64,4,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>(&groupgemm_hz_fused_4,"groupgemm_hz_fused_4");}};static Register_groupgemm_hz_fused_4 register_groupgemm_hz_fused_4;
} // namespace mxmoe
