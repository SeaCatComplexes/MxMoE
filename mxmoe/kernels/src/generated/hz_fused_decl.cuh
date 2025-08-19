
#pragma once

#include "cta_gemm.cuh"
#include "registry.cuh"

namespace mxmoe {

using namespace tiled_gemm;
using MatricesInfo = RCR_FP16FP16FP16;


// template<TileConfig<128,128,64,2,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
__global__ void groupgemm_hz_fused_0_impl (     //
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
) ;




// template<TileConfig<128,128,64,2,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
void groupgemm_hz_fused_0(                      //
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
) ;

/////////////////////////////////////////////////////


// template<TileConfig<128,128,32,2,2,1,4,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
__global__ void groupgemm_hz_fused_1_impl (     //
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
) ;




// template<TileConfig<128,128,32,2,2,1,4,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
void groupgemm_hz_fused_1(                      //
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
) ;

/////////////////////////////////////////////////////


// template<TileConfig<128,128,32,2,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
__global__ void groupgemm_hz_fused_2_impl (     //
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
) ;




// template<TileConfig<128,128,32,2,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
void groupgemm_hz_fused_2(                      //
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
) ;

/////////////////////////////////////////////////////


// template<TileConfig<128,128,32,2,4,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
__global__ void groupgemm_hz_fused_3_impl (     //
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
) ;




// template<TileConfig<128,128,32,2,4,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
void groupgemm_hz_fused_3(                      //
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
) ;

/////////////////////////////////////////////////////


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
) ;




// template<TileConfig<128,128,64,4,2,1,3,-1,MMA_FP16_FP32,NO_QUANT,NO_QUANT>>
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
) ;

/////////////////////////////////////////////////////


}  // namespace mxmoe
