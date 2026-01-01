# ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs


![arcquant](kernels/ARCQuant_Visualize.png)

**ARCQuant** is a high-performance quantization framework designed to resolve the conflict between **accuracy** and **inference efficiency** in low-bit LLMs.

While fine-grained quantization (e.g., Block-wise/NVFP4) effectively isolates quantization noise, **activation outliers** still degrade performance in critical channels. Traditional mixed-precision methods address this by splitting computations into separate branches, which introduces significant kernel launch overhead and memory fragmentation.

**ARCQuant takes a different approach.** Instead of treating outliers separately, we leverage the **structural sparsity of quantization errors** in fine-grained settings. We capture the quantization residuals of these critical channels and fuse them back into the computation as **Augmented Residual Channels (ARC)**.

### 🚀 Key Features

*   **Unified Single-Kernel Execution:** By converting error compensation into channel augmentation, ARCQuant performs the entire inference using a **single, standard GEMM kernel**. This decouples the algorithm from complex custom operators and allows full utilization of optimized libraries like **CUTLASS**.
*   **Accuracy-Aware Compensation:** Powered by a rigorous analysis of error bounds, ARCQuant identifies and compensates only the most critical "heavy-hitter" channels, recovering **FP16-level accuracy** with negligible computational cost.
*   **Hardware-Friendly Design:** Designed for modern GPU architectures, ARCQuant eliminates the bottleneck of integer dequantization on CUDA cores, making it a future-proof solution for native floating-point quantization.

### 📊 Performance
On Llama-3 and Qwen-2.5 models, ARCQuant achieves **state-of-the-art accuracy** while delivering significantly lower latency compared to traditional mixed-precision baselines.


## 1. Installation
```bash
conda create -n arcquant python=3.10 -y
conda activate arcquant
```
Please make sure that [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is in your environment.
```bash
git clone --recurse-submodules https://github.com/actypedef/ARCQuant.git
cd ARCQuant
pip install -r requirements.txt
```

## 2. Usage

### 2.1 Building Kernels
```bash
sudo apt-get update
sudo apt-get install python3-dev
```
```bash
conda install pybind11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
```bash
cd kernels/
bash remake.sh
```

This might take a few minutes.

### 2.2 Preprocessing

Reorder_indices, select_num are needed for quantization:
```bash
python reorder_indices.py --model /PATH/TO/YOUR/MODEL/ --samples 128 --seqlen 2048 --act_sort_metric max
```
Results are saved in ./saved/

### 2.3 Accuracy Evaluation
```bash
bash run_arcquant.sh /PATH/TO/YOUR/MODEL/
```

## 3. Efficiency Evaluation

End-to-end efficiency:
```bash
python benchmarks/benchmark_e2e_arc.py --model 'llama-2-7b' --batch_size 8 --prefill_seq_len 1024 --decode_steps 50
```
TensorRT efficiency:
```bash
pip install tensorrt
python benchmarks/trt-fp8-prefill-llama.py
```
