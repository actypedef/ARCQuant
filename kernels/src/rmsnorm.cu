#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "reorder.cuh"
#include "cutlass/numeric_conversion.h"

#include <cstdio>


#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

#define FP4_MAX 6
#define FP8_MAX 448
#define SCALE_EPS 0.001953125

typedef cutlass::float_e2m1_t fp4_t;
typedef cutlass::float_ue4m3_t sf_t;
typedef cutlass::bfloat16_t bf16_t;

namespace cg = cooperative_groups;
using namespace cute;

struct PackFp4 {
  int8_t low : 4;
  int8_t high : 4;
};

HOST_DEVICE float fpmax(float a, float b) { return (a) > (b) ? (a) : (b); }

HOST_DEVICE float fpmin(float a, float b) { return (a) < (b) ? (a) : (b); }

HOST_DEVICE float clamp(float x, float a, float b) { return fpmax(a, fpmin(b, x)); }

template <typename T> HOST_DEVICE T abs(T x) { return x < (T)0 ? -x : x; }

template <typename T, typename U, typename Accum, int Size = sizeof(U) / sizeof(T)>
HOST_DEVICE Accum local_sum_p2(U *vec, Accum sumv) {
  T *view = reinterpret_cast<T *>(vec);
  #pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    sumv += (Accum)view[i] * (Accum)view[i];
  }
  return sumv;
}

#define GROUP_NUM(x) ((x) / 16)

#define mymax(a, b) ((a) > (b) ? (a) : (b))

template <typename T, typename U, int Size = sizeof(U) / sizeof(T)>
DEVICE float local_abs_max(U *vec, float maxv) {
  T *view = reinterpret_cast<T *>(vec);
  #pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    maxv = mymax((float)maxv, (float)abs((float)view[i]));
  }
  return maxv;
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Activation Reorder Quantize /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void rmsnorm_x_kernel(
  bf16_t *input,
  bf16_t *weight,
  float eps,
  int16_t *reorder_index,
  uint8_t *q_out,
  auto q_scale_tensor,
  int KQ, int KE
){
  constexpr int elements_per_thread = GROUP_SIZE;

  cg::thread_block cta = cg::this_thread_block();

  // One block solves one row of hidden states.
  __shared__ uint8_t smem[HIDDEN_DIM * sizeof(bf16_t) + bdx * sizeof(float)];
  bf16_t *input_smem = reinterpret_cast<bf16_t*>(smem);

  constexpr int input_smem_bytes = HIDDEN_DIM * sizeof(bfloat16_t);
  float *sum_smem = reinterpret_cast<float *>(smem + input_smem_bytes);

  // Local memory stores the reordered hidden states.
  bf16_t input_frag[elements_per_thread];
  bf16_t output_frag[elements_per_thread / 2];

  // Row are independent
  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;
  q_out = q_out + row_id * (GROUP_SIZE * GROUP_NUM(KQ + KE)) / 2;

  // Coalesced access global memory
  int tx = threadIdx.x;
  int tid = tx;
  constexpr int bytes_per_iter = bdx * 16;
  constexpr int iters = HIDDEN_DIM * sizeof(bf16_t) / bytes_per_iter;
  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> Float2E2m1;
  cutlass::NumericConverter<float, fp4_t, cutlass::FloatRoundStyle::round_to_nearest> E2m12Float;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> Float2Ue4m3;
  cutlass::NumericConverter<float, sf_t, cutlass::FloatRoundStyle::round_to_nearest> Ue4m32Float;
  cutlass::NumericConverter<bf16_t, int, cutlass::FloatRoundStyle::round_to_nearest> Int2Bfloat16;
  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> Float2Bfloat16;
  cutlass::NumericConverter<int, fp4_t, cutlass::FloatRoundStyle::round_to_nearest> E2m12Int;

  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);

  #pragma unroll
  for(int i = 0;i < iters;++i){
    // Each thread loads 16 bytes
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = input_frag_float4[i] = *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  
  float sumv = 0.f;
    
#pragma unroll
  for(int i = 0;i < iters;++i){
    sumv = local_sum_p2<bf16_t, float4, float>(input_frag_float4 + i, sumv);
  }

  sum_smem[tid] = sumv;
  cta.sync();

  // Block-level reduction. Using shared memory.
  if(tid < 256) {
    sum_smem[tid] = sumv = sumv + ((tid + 256) < bdx ? sum_smem[tid + 256] : 0);
  }
  cta.sync();
  if(tid < 128) {
    sum_smem[tid] = sumv = sumv + ((tid + 128) < bdx ? sum_smem[tid + 128] : 0);
  }
  cta.sync();
  if(tid < 64) {
    sum_smem[tid] = sumv = sumv + sum_smem[tid + 64];
  }
  cta.sync();
  if(tid < 32) {
    sum_smem[tid] = sumv = sumv + sum_smem[tid + 32];
  }
  cta.sync();

  // Within single warp
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  if (tid < 32) {
    for (int s = 16; s > 0; s >>= 1) {
      sumv += tile32.shfl_down(sumv, s);
    }
  }
  // this makes sumv equal to rvariance
  if (tid == 0) {
    sumv = rsqrt(sumv / HIDDEN_DIM + eps);
    sum_smem[0] = sumv;
  }
  // broadcast the rvariance of tid 0 to all the threads
  cta.sync();
  sumv = sum_smem[0];

  // Reorder
  #pragma unroll 4
  for(int i = 0;i < elements_per_thread;++i){
    int reorder_idx = (int32_t)reorder_index[tid * GROUP_SIZE + i]; 
    bf16_t input_value = input_smem[reorder_idx];
    bf16_t weight_value = weight[reorder_idx];
    input_frag[i] = Float2Bfloat16((float)input_value * (float)weight_value * (float)sumv);
  }
  // Reduce to get max
  // Each ty should get its max value
  float2 *input_frag_float2 = reinterpret_cast<float2 *>(input_frag);
  constexpr int float4_per_thread = elements_per_thread * sizeof(bf16_t) / sizeof(float4);
  float maxv = 0,  scale = 1.0, r_scale = 1.0;

  #pragma unroll
  for(int i = 0; i < float4_per_thread;++i){
    maxv = local_abs_max<bf16_t, float4>(input_frag_float4 + i, maxv);
  }
  cta.sync();

  // Calculate scales
  // Specific layout
  float lower_bound, upper_bound;
  // Q quantize
  lower_bound = -FP4_MAX;
  upper_bound = FP4_MAX;
  scale = clamp(maxv / FP4_MAX, SCALE_EPS, FP8_MAX);
  int pos = tid + mymax(0, tid - GROUP_NUM(KQ - KE));
  auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto logical_coord1 = make_coord(make_coord(0, pos % 4), pos / 4);
  auto logical_coord2 = make_coord(0, 0);
  q_scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = Float2Ue4m3(scale);

  // Use reverse scale to replace devision by multiplication
   r_scale = 1.0 / Ue4m32Float(Float2Ue4m3(scale));

  // Quantize each thread's value
  // Each iteration quantize two things, convenient for packing int4
  PackFp4* output_frag_fp4 = reinterpret_cast<PackFp4*>(output_frag);
  for(int i = 0; i < elements_per_thread; i += 4){
    float result_0, result_1, result_2, result_3;
    result_0 = clamp(((float)input_frag[i + 0] * r_scale), lower_bound, upper_bound);
    result_1 = clamp(((float)input_frag[i + 1] * r_scale), lower_bound, upper_bound);
    result_2 = clamp(((float)input_frag[i + 2] * r_scale), lower_bound, upper_bound);
    result_3 = clamp(((float)input_frag[i + 3] * r_scale), lower_bound, upper_bound);
    input_frag[i + 0] = Float2Bfloat16((float)input_frag[i + 0] - E2m12Float(Float2E2m1(result_0)) * Ue4m32Float(Float2Ue4m3(scale)));
    input_frag[i + 1] = Float2Bfloat16((float)input_frag[i + 1] - E2m12Float(Float2E2m1(result_1)) * Ue4m32Float(Float2Ue4m3(scale)));
    input_frag[i + 2] = Float2Bfloat16((float)input_frag[i + 2] - E2m12Float(Float2E2m1(result_2)) * Ue4m32Float(Float2Ue4m3(scale)));
    input_frag[i + 3] = Float2Bfloat16((float)input_frag[i + 3] - E2m12Float(Float2E2m1(result_3)) * Ue4m32Float(Float2Ue4m3(scale)));
    output_frag_fp4[i / 2 + 0].low = Float2E2m1(result_0).storage;
    output_frag_fp4[i / 2 + 0].high = Float2E2m1(result_1).storage;
    output_frag_fp4[i / 2 + 1].low = Float2E2m1(result_2).storage;
    output_frag_fp4[i / 2 + 1].high = Float2E2m1(result_3).storage;
  }
  const int ke_thread_count = GROUP_NUM(KE);
  const int kq_thread_count = bdx - ke_thread_count;
  if(tid >= bdx - GROUP_NUM(KE)){
    maxv = 0;
    #pragma unroll
    for(int i = 0; i < float4_per_thread;++i){
      maxv = local_abs_max<bf16_t, float4>(input_frag_float4 + i, maxv);
    }
    scale = clamp(maxv / FP4_MAX, SCALE_EPS, FP8_MAX);
    logical_coord1 = make_coord(make_coord(0, (pos + 1) % 4), (pos + 1) / 4);
    q_scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = Float2Ue4m3(scale);

    r_scale = 1.0 / Ue4m32Float(Float2Ue4m3(scale));
    int q_offset = elements_per_thread / 2;
    for(int i = 0; i < elements_per_thread; i += 4){
      float result_0, result_1, result_2, result_3;
      result_0 = clamp(((float)input_frag[i + 0] * r_scale), lower_bound, upper_bound);
      result_1 = clamp(((float)input_frag[i + 1] * r_scale), lower_bound, upper_bound);
      result_2 = clamp(((float)input_frag[i + 2] * r_scale), lower_bound, upper_bound);
      result_3 = clamp(((float)input_frag[i + 3] * r_scale), lower_bound, upper_bound);
      output_frag_fp4[i / 2 + q_offset + 0].low = Float2E2m1(result_0).storage;
      output_frag_fp4[i / 2 + q_offset + 0].high = Float2E2m1(result_1).storage;
      output_frag_fp4[i / 2 + q_offset + 1].low = Float2E2m1(result_2).storage;
      output_frag_fp4[i / 2 + q_offset + 1].high = Float2E2m1(result_3).storage;
    }
    
    const int kq_region_bytes = kq_thread_count * 8;
    const int ke_thread_idx = tid - kq_thread_count;
    const int ke_thread_offset = kq_region_bytes + ke_thread_idx * 16;
    
    float4* q_out_ptr = reinterpret_cast<float4*>(q_out + ke_thread_offset);
    *q_out_ptr = *(reinterpret_cast<float4*>(output_frag));
  }
  else{
    float2* q_out_ptr = reinterpret_cast<float2*>(q_out + tid * 8);
    *q_out_ptr = *(reinterpret_cast<float2*>(output_frag));
  }
}


template<int group_size, int hidden_dim>
void run_rmsnorm_x_bf16_nvfp4(
  bf16_t *hidden_states,
  bf16_t *weight,
  float eps,
  int seq_len,
  int16_t *reorder_index,
  uint8_t *q_out,
  sf_t *q_scale,
  int KQ, int KE
){
  dim3 grids(seq_len);
  dim3 blocks(hidden_dim / group_size);
  Tensor q_scale_tensor = cute::make_tensor(q_scale, filter_zeros(nvfp4::get_layoutSFA(seq_len, KQ + KE)));
  rmsnorm_x_kernel<hidden_dim / group_size, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (bf16_t *)weight,
    eps,
    (int16_t *)reorder_index,
    (uint8_t *)q_out,
    q_scale_tensor,
    KQ, KE
  );
}

///////////////////////////////// Llama /////////////////////////////////
template void run_rmsnorm_x_bf16_nvfp4<16, 4096>(
  bf16_t*, bf16_t*, float, int, int16_t*, uint8_t*,
  sf_t*, int, int
);

template void run_rmsnorm_x_bf16_nvfp4<16, 8192>(
  bf16_t*, bf16_t*, float, int, int16_t*, uint8_t*,
  sf_t*, int, int
);

///////////////////////////////// Qwen /////////////////////////////////
template void run_rmsnorm_x_bf16_nvfp4<16, 5120>(
  bf16_t*, bf16_t*, float, int, int16_t*, uint8_t*,
  sf_t*, int, int
);

template void run_rmsnorm_x_bf16_nvfp4<16, 3584>(
  bf16_t*, bf16_t*, float, int, int16_t*, uint8_t*,
  sf_t*, int, int
);
