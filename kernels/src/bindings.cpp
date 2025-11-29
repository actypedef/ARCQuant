#include <torch/extension.h>
#include <iostream>

// Include all files
#include "nvfp4.h"
#include "reorder.cuh"
#include <flashinfer.h>

inline size_t get_sfa_buffer_size_in_bytes(int num_rows, int K_dim) {
    auto layout = filter_zeros(nvfp4::get_layoutSFA(num_rows, K_dim));
    size_t num_elements = cute::size(layout);
    // return num_elements * sizeof(cutlass::float_ue4m3_t);
    return (num_rows / 128 + 1) * 128 * K_dim / 16;
}

inline size_t get_sfb_buffer_size_in_bytes(int num_rows, int K_dim) {
    auto layout = filter_zeros(nvfp4::get_layoutSFB(num_rows, K_dim));
    size_t num_elements = cute::size(layout);
    // return num_elements * sizeof(cutlass::float_ue4m3_t);
    return (num_rows / 128 + 1) * 128 * K_dim / 16;
}



torch::Tensor matmul(
        const torch::Tensor &A,
        const torch::Tensor &B,
        const torch::Tensor &SFA,
        const torch::Tensor &SFB,
        const float scale
)
{
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * 2;  // 4bit packing is on the columns
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host_nvfp4_bf16(
        reinterpret_cast<cutlass::float_e2m1_t *>(A.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_e2m1_t *>(B.data_ptr<uint8_t>()),
        M, N, K,
        (cutlass::bfloat16_t *)C.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)C.data_ptr<at::BFloat16>(),
        reinterpret_cast<cutlass::float_ue4m3_t *>(SFA.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_ue4m3_t *>(SFB.data_ptr<uint8_t>()),
        scale
    );
    return C;
}

std::tuple<torch::Tensor, torch::Tensor> reorder_quantize_x(
        const torch::Tensor &X,
        const torch::Tensor &reorder_index,
        const int KE
)
{
    int M = X.size(0);
    int KQ = X.size(1);
    int K = KQ + KE;
    auto QX = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFX = torch::empty({(int)get_sfa_buffer_size_in_bytes(M, K)}, torch::dtype(torch::kUInt8).device(X.device()));
    if (KQ == 4096) { // Llama
        run_reorder_x_bf16_nvfp4<16, 4096>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 8192) { // Llama
        run_reorder_x_bf16_nvfp4<16, 8192>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 14336) {
        run_reorder_x_bf16_nvfp4<16, 14336>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 11008) {
        run_reorder_x_bf16_nvfp4<16, 11008>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 5120) { // Qwen
        run_reorder_x_bf16_nvfp4<16, 5120>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 13824) {
        run_reorder_x_bf16_nvfp4<16, 13824>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 3584) { // Qwen
        run_reorder32_x_bf16_nvfp4<32, 3584>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 18944) {
        run_reorder32_x_bf16_nvfp4<32, 18944>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 27648) {
        run_down32_x_bf16_nvfp4<32, 27648>(
            (cutlass::bfloat16_t *)(torch::index_select(X, 1, reorder_index.to(torch::kInt32))).data_ptr<at::BFloat16>(), M, 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 28672) {
        run_down32_x_bf16_nvfp4<32, 28672>(
            (cutlass::bfloat16_t *)(torch::index_select(X, 1, reorder_index.to(torch::kInt32))).data_ptr<at::BFloat16>(), M, 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_x_bf16_nvfp4 "));
    }
    return std::make_tuple(QX, SFX);
}

std::tuple<torch::Tensor, torch::Tensor> reorder_quantize_w(
        const torch::Tensor &W,
        const torch::Tensor &reorder_index,
        const int KE
)
{
    int N = W.size(0);
    int KQ = W.size(1);
    int K = KQ + KE;
    auto QW = torch::empty({N, K / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFW = torch::empty({(int)get_sfb_buffer_size_in_bytes(N, K)}, torch::dtype(torch::kUInt8).device(W.device()));
    if (KQ == 4096) { //Llama
        run_reorder_w_bf16_nvfp4<16, 4096>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 8192) { //Llama
        run_reorder_w_bf16_nvfp4<16, 8192>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 14336) {
        run_reorder_w_bf16_nvfp4<16, 14336>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 11008) {
        run_reorder_w_bf16_nvfp4<16, 11008>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 5120) { //Qwen
        run_reorder_w_bf16_nvfp4<16, 5120>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 13824) {
        run_reorder_w_bf16_nvfp4<16, 13824>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 3584) { //Qwen
        run_reorder32_w_bf16_nvfp4<32, 3584>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 18944) {
        run_reorder32_w_bf16_nvfp4<32, 18944>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 27648) {
        run_down32_w_bf16_nvfp4<32, 27648>(
            (cutlass::bfloat16_t *)(torch::index_select(W, 1, reorder_index.to(torch::kInt32))).data_ptr<at::BFloat16>(), N, 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 28672) {
        run_down32_w_bf16_nvfp4<32, 28672>(
            (cutlass::bfloat16_t *)(torch::index_select(W, 1, reorder_index.to(torch::kInt32))).data_ptr<at::BFloat16>(), N, 
            QW.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_w_bf16_nvfp4 "));
    }
    return std::make_tuple(QW, SFW);
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_quantize_x(
        const torch::Tensor &X,
        const torch::Tensor &W,
        const float eps,
        const torch::Tensor &reorder_index,
        const int KE
)
{
    int M = X.size(0);
    int KQ = X.size(1);
    int K = KQ + KE;
    auto QX = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFX = torch::empty({(int)get_sfa_buffer_size_in_bytes(M, K)}, torch::dtype(torch::kUInt8).device(X.device()));
    if (KQ == 4096) { // Llama
        run_rmsnorm_x_bf16_nvfp4<16, 4096>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps, 
            M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 8192) { // Llama
        run_rmsnorm_x_bf16_nvfp4<16, 8192>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps, 
            M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 5120) { // Qwen
        run_rmsnorm_x_bf16_nvfp4<16, 5120>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps, 
            M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else if (KQ == 3584) { // Qwen
        run_rmsnorm_x_bf16_nvfp4<16, 3584>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps, 
            M, reorder_index.data_ptr<int16_t>(), 
            QX.data_ptr<uint8_t>(), reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>()), 
            KQ, KE
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_rmsnorm_x_bf16_nvfp4 "));
    }
    return std::make_tuple(QX, SFX);
}

// ===== Flash Infer ======
inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)


void batch_decode_i4(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  FlashInferBatchDecodeKernel_i4<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void init_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void batch_decode_f16(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  FlashInferBatchDecodeKernel_f16<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void init_kv_f16(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_f16<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_f16(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_f16<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}



//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("matmul", &matmul,
          "nvfp4 matmul, return bfloat16 tensor", 
          py::arg("A"), py::arg("B"),
          py::arg("SFA"), py::arg("SFB"), py::arg("scale")
        );
    m.def("reorder_quantize_x", &reorder_quantize_x,
          "Reorder and quantize activation, return K + KE channels",
          py::arg("X"), py::arg("reorder_index"),
          py::arg("KE")
        );
    m.def("reorder_quantize_w", &reorder_quantize_w,
          "Reorder and quantize weight, return K + KE channels",
          py::arg("W"), py::arg("reorder_index"),
          py::arg("KE")
        );
    m.def("rmsnorm_quantize_x", &rmsnorm_quantize_x,
          "Rmsnorm and quantize activation, return K + KE channels",
          py::arg("X"), py::arg("W"), py::arg("eps"), 
          py::arg("reorder_index"),
          py::arg("KE")
        );
    m.def("batch_decode_i4", &batch_decode_i4, "");
    m.def("init_kv_i4", &init_kv_i4, "");
    m.def("append_kv_i4", &append_kv_i4, "");
    m.def("batch_decode_f16", &batch_decode_f16, "");
    m.def("init_kv_f16", &init_kv_f16, "");
    m.def("append_kv_f16", &append_kv_f16, ""); 
}