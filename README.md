<p align="center">
<img src="assets/deepcompressor.png" alt="DeepCompressor Logo" width="450">
</p>

<h2><p align="center">Model Compression Toolbox for Large Language Models and Diffusion Models</p></h2>

<p align="center">
    <a href="https://github.com/mit-han-lab/deepcompressor/blob/master/LICENSE">
        <img alt="Apache License" src="https://img.shields.io/github/license/mit-han-lab/deepcompressor">
    </a>
    <!-- <a href="https://deepcompressor.mit.edu">
        <img alt="Website" src="https://img.shields.io/website?up_message=deepcompressor&url=https%3A%2F%2Fdeepcompressor.mit.edu">
    </a> -->
   <!-- <a href="https://pypi.org/project/deepcompressor/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/deepcompressor">
    </a> -->
</p>
<br />


## News
- [2024/10] 🔥 Our latest **W4A4** Diffusion model quantization work **SVDQuant** algorithm and **Nunchaku** system is pubicly released! Check our [paper](http://arxiv.org/abs/2411.05007)!
- [2024/05] 🔥 Our latest **W4A8KV4** LLM quantization work **QoQ** algorithm and **QServe** system is publicly released! **QoQ** is short for *quattuor-octō-quattuor* which is 4-8-4 in latin. Check our [paper](https://arxiv.org/abs/2405.04532)!

## Key Features

***DeepCompressor*** is an open source model compression toolbox for large language models and diffusion models based on PyTorch. DeepCompressor currently supports fake quantization with any integer and floating-point data type within 8 bits, e.g., INT8, INT4 and FP4_E2M1. Here are examples that implement the following algorithms.
+ [Post-training quantization for large language models](/examples/llm/):
  + Weight-only Quantization
    + [AWQ (W4A16)](/examples/llm/configs/awq.yaml)
    + [GPTQ (W4A16)](/examples/llm/configs/gptq.yaml)
  + Weight-Activation Quantization
    + [SmoothQuant (W8A8)](/examples/llm/configs/smoothquant-static.yaml)
  + Weight-Activation and KV-Cache Quantization
    + [QoQ (W4A8KV4)](/examples/llm/)
+ [Post-training quantization for diffusion models](/examples/diffusion/):
  + Weight-Activation Quantization
    + [SVDQuant (W4A4)](/examples/diffusion/)

DeepCompressor also contains examples that integrate with other inference libraries.
  + [Deploy weight-only quantized LLMs with TinyChat](/examples/llm/)
  + [Deploy quantized LLMs with QServe]((/examples/llm/))
  + [Deploy quantized diffusion models with Nunchaku](/examples/diffusion/)

## Installation

### Install from Source

1. Clone this repository and navigate to deepcompressor folder
```
git clone https://github.com/mit-han-lab/deepcompressor
cd deepcompressor
```

2. Install Package
```
conda env create -f environment.yml
poetry install
```

## Highlights
### SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models

[[Website](https://hanlab.mit.edu/projects/svdquant)][[Paper](http://arxiv.org/abs/2411.05007)][[Nunchaku Inference System](https://github.com/mit-han-lab/nunchaku)]

Diffusion models have been proven highly effective at generating high-quality images. However, as these models grow larger, they require significantly more memory and suffer from higher latency, posing substantial challenges for deployment. In this work, we aim to accelerate diffusion models by quantizing their weights and activations to 4 bits. At such an aggressive level, both weights and activations are highly sensitive to quantization, where conventional post-training quantization methods for large language models like smoothing become insufficient. To overcome this limitation, we propose **SVDQuant**, a new 4-bit quantization paradigm. Different from smoothing which redistributes outliers between weights and activations, our approach *absorbs* these outliers using a low-rank branch. We first shift the outliers from activations into the weights, then employ a high-precision low-rank branch to take in the outliers in the weights with SVD. This process eases the quantization on both sides. However, naively running the low-rank branch independently incurs significant overhead due to extra data movement of activations, negating the quantization speedup. To address this, we co-design an inference engine **Nunchaku** that fuses the kernels in the low-rank branch into thosein the low-bit branch to cut off redundant memory access. It can also seamlessly support off-the-shelf low-rank adapters (LoRAs) without the requantization. Extensive experiments on SDXL, PixArt-Sigma, and FLUX.1 validate the effectiveness of SVDQuant in preserving image quality. We reduce the memory usage for the 12B FLUX.1 models by 3.6×, achieving 3.5× speedup over the 4-bit weight-only quantized baseline on a 16GB RTX-4090 GPU, paving the way for more interactive applications on PCs.

![Teaser](/assets/diffusion/svdquant/teaser.jpg)
![SVDQuant](/assets/diffusion/svdquant/svdquant.png)

#### Quality Evaluation

Below is the quality and similarity evaluated with 5000 samples from MJHQ-30K dataset. IR means ImageReward. Our 4-bit results outperform other 4-bit baselines, effectively preserving the visual quality of 16-bit models.

| Model                      | Precision | Method  | FID ($\downarrow$) | IR ($\uparrow$) | LPIPS ($\downarrow$) | PSNR( $\uparrow$) |
|----------------------------|-----------|---------|--------------------|-----------------|----------------------|-------------------|
| FLUX.1-dev (50 Steps)      | BF16      | --      | 20.3               | 0.953           | --                   | --                |
|                            | INT W8A8  | Ours    | 20.4               | 0.948           | 0.089                | 27.0              |
|                            | W4A16     | NF4     | 20.6               | 0.910           | 0.272                | 19.5              |
|                            | INT W4A4  | Ours    | **19.86**          | 0.932           | 0.254                | 20.1              |
|                            | FP W4A4   | Ours    | 21.0               | **0.933**       | **0.247**            | **20.2**          |
| FLUX.1-schnell (4 Steps)   | BF16      | --      | 19.2               | 0.938           | --                   | --                |
|                            | INT W8A8  | Ours    | 19.2               | 0.966           | 0.120                | 22.9              |
|                            | W4A16     | NF4     | 18.9               | 0.943           | 0.257                | 18.2              |
|                            | INT W4A4  | Ours    | **18.4**           | **0.969**       | 0.292                | 17.5              |
|                            | FP W4A4   | Ours    | 19.9               | 0.956           | 0.279                | 17.5              |
|                            | FP16      | --      | 16.6               | 0.944           | --                   | --                |
| PixArt-Sigma (20 Steps)    | INT W8A8  | ViDiT-Q | 15.7               | 0.944           | 0.137                | 22.5              |
|                            | INT W8A8  | Ours    | 16.3               | **0.955**       | **0.109**            | **23.7**          |
|                            | INT W4A8  | ViDiT-Q | 37.3               | 0.573           | 0.611                | 12.0              |
|                            | INT W4A4  | Ours    | 20.1               | 0.898           | 0.394                | 16.2              |
|                            | FP W4A4   | Ours    | **18.3**           | **0.946**       | **0.326**            | **17.4**          |

### QServe: W4A8KV4 Quantization for Efficient LLM Serving

[[Website](https://hanlab.mit.edu/projects/qserve)][[Paper](https://arxiv.org/abs/2405.04532)][[QoQ Algorithm Code](/examples/llm)][[QServe GPU System](https://github.com/mit-han-lab/qserve)]

Quantization can accelerate large language model (LLM) inference. Going beyond INT8 quantization, the research community is actively exploring even lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization techniques only accelerate low-batch, edge LLM inference, failing to deliver performance gains in large-batch, cloud-based LLM serving. We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when **dequantizing either weights or partial sums** on GPUs. To address this challenge, we introduce **QoQ**, a W4A8KV4 quantization algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands for **quattuor-octo-quattuor**, which represents 4-8-4 in Latin. QoQ is implemented by the **QServe** inference library that achieves measured speedup. The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by **operations on low-throughput CUDA cores**. Building upon this insight, in QoQ algorithm, we introduce progressive quantization that can allow low dequantization overhead in W4A8 GEMM. Additionally, we develop SmoothAttention to effectively mitigate the accuracy degradation incurred by 4-bit KV quantization. In the QServe system, we perform compute-aware weight reordering and take advantage of register-level parallelism to reduce dequantization latency. We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization. As a result, QServe improves the maximum achievable serving throughput of Llama-3-8B by **1.2×** on A100, **1.4×** on L40S; and Qwen1.5-72B by **2.4×** on A100, **3.5×** on L40S, compared to TensorRT-LLM.

![QoQ-QServe](/assets/llm/qoq/qoq-qserve.png)
![QoQ](/assets/llm/qoq/qoq.png)


#### Perplexity Evaluation

Below is the WikiText2 perplexity evaluated with 2048 sequence length. The lower is the better.

|   Methods   |  Precision   | Llama-3.1 70B | Llama-3.1 8B | Llama-3 70B |  Llama-3 8B | Llama-2 7B | Llama-2 13B | Llama-2 70B | Llama 7B | Llama 13B | Llama 30B | Mistral 7B | Yi 34B |
|-------------|--------------|---------------|--------------|-------------| ------------|------------|-------------|-------------|----------|-----------|-----------|------------|--------|
| FP16        |              | 2.81          | 6.24         | 2.85        |  6.14       | 5.47       | 4.88        | 3.32        | 5.68     | 5.09      | 4.10      | 5.25       | 4.60   |
| SmoothQuant | W8A8         | 3.23          | 6.38         | 3.14        |  6.28       | 5.54       | 4.95        | 3.36        | 5.73     | 5.13      | 4.23      | 5.29       | 4.69   |
| GPTQ-R      | W4A16 g128   | 3.46          | 6.64         | 3.42        |  6.56       | 5.63       | 4.99        | 3.43        | 5.83     | 5.20      | 4.22      | 5.39       | 4.68   |
| AWQ         | W4A16 g128   | 3.22          | 6.60         | 3.20        |  6.54       | 5.60       | 4.97        | 3.41        | 5.78     | 5.19      | 4.21      | 5.37       | 4.67   |
| QuaRot      | W4A4         | 5.97          | 8.32         | 6.75        |  8.33       | 6.19       | 5.45        | 3.83        | 6.34     | 5.58      | 4.64      | 5.77       | NaN    |
| Atom        | W4A4 g128    | -             | -            | 4.33        |  7.78       | 6.12       | 5.31        | 3.73        | 6.25     | 5.52      | 4.61      | 5.76       | 4.97   |
| QoQ         | W4A8KV4      | 3.69          | 6.91         | 3.65        |  6.84       | 5.75       | 5.11        | 3.51        | 5.92     | 5.27      | 4.32      | 5.45       | 4.73   |
| QoQ         | W4A8KV4 g128 | 3.54          | 6.80         | 3.51        |  6.73       | 5.68       | 5.05        | 3.46        | 5.88     | 5.23      | 4.27      | 5.41       | 4.73   |

\* SmoothQuant is evaluated with per-tensor static KV cache quantization.

#### Efficiency Benchmarks

When serving the large language models Llama-3-8B and Qwen1.5-72B on L40S and A100 GPUs, QServe demonstrates superior performance, achieving **1.2x-1.4x higher throughput** compared to the leading industry solution, TensorRT-LLM, for Llama-3-8B, and a **2.4x-3.5x higher throughput** for Qwen1.5-72B.

See more about benchmarking setting in [QServe GPU Inference System](https://github.com/mit-han-lab/qserve).
| L40S (48G)           | Llama-3-8B | Llama-2-7B | Mistral-7B | Llama-2-13B | Llama-30B | Yi-34B    | Llama-2-70B | Qwen-1.5-72B |
|----------------------|------------|------------|------------|-------------|-----------|-----------|-------------|--------------|
| TRT-LLM-FP16         | 1326       | 444        | 1566       | 92          | OOM       | OOM       | OOM         | OOM          |
| TRT-LLM-W4A16        | 1431       | 681        | 1457       | 368         | 148       | 313       | 119         | 17           |
| TRT-LLM-W8A8         | 2634       | 1271       | 2569       | 440         | 123       | 364       | OOM         | OOM          |
| Atom-W4A4            | --         | 2120       | --         | --          | --        | --        | --          | --           |
| QuaRot-W4A4          | --         | 805        | --         | 413         | 133       | --        | --          | 15           |
| QServe-W4A8KV4       | **3656**   | **2394**   | **3774**   | **1327**    | **504**   | **869**   | **286**     | **59**       |
| Throughput Increase* | **1.39x**  | **1.13x**  | **1.47x**  | **3.02x**   | **3.41x** | **2.39x** | **2.40x**   | **3.47x**    |

| A100 (80G)           | Llama-3-8B | Llama-2-7B | Mistral-7B | Llama-2-13B | Llama-30B | Yi-34B    | Llama-2-70B | Qwen-1.5-72B |
|----------------------|------------| -----------|------------|-------------|-----------|-----------|-------------|--------------|
| TRT-LLM-FP16         | 2503       | 1549       | 2371       | 488         | 80        | 145       | OOM         | OOM          |
| TRT-LLM-W4A16        | 2370       | 1549       | 2403       | 871         | 352       | 569       | 358         | 143          |
| TRT-LLM-W8A8         | 2396       | 2334       | 2427       | 1277        | 361       | 649       | 235         | 53           |
| Atom-W4A4            | --         | 1160       | --         | --          | --        | --        | --          | --           |
| QuaRot-W4A4          | --         | 1370       | --         | 289         | 267       | --        | --          | 68           |
| QServe-W4A8KV4       | **3005**   | **2908**   | **2970**   | **1741**    | **749**   | **803**   | **419**     | **340**      |
| Throughput Increase* | **1.20x**  | **1.25x**  | **1.22x**  | **1.36x**   | **2.07x** | **1.23x** | **1.17x**   | **2.38x**    |

The absolute token generation throughputs of QServe and baseline systems (Unit: tokens/second. `--` means unsupported). All experiments were conducted under the same device memory budget. Throughput increase of QServe is calculated with regard to the best baseline in each column.

## Reference

If you find `deepcompressor` useful or relevant to your research, please kindly cite our paper:

```
@article{lin2024qserve,
  title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
  author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
  journal={arXiv preprint arXiv:2405.04532},
  year={2024}
}
```

## Related Projects

The following projects are highly related to QServe. Our group has developed full-stack application-algorithm-system-hardware support for efficient large models, receiving **9k+ GitHub stars** and **over 1M Huggingface community downloads**.

You are also welcome to check out [MIT HAN LAB](https://hanlab.mit.edu) for other exciting projects on **Efficient Generative AI**!

- [**System**] [QServe: W4A8KV4 Quantization for Efficient LLM Serving](https://github.com/mit-han-lab/qserve)

- [**System**] [TinyChat: Efficient and Lightweight Chatbot with AWQ](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat)

- [**Application**] [VILA: On Pretraining of Visual-Language Models](https://github.com/Efficient-Large-Model/VILA)

- [**Algorithm**] [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

- [**Algorithm**] [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

- [**Algorithm**] [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://github.com/mit-han-lab/distrifuser)

- [**Hardware**] [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning](https://arxiv.org/abs/2012.09852)


## Acknowledgement

DeepCompressor is inspired by many open-source libraries, including (but not limited to) [GPTQ](https://arxiv.org/abs/2210.17323), [QuaRot](https://arxiv.org/abs/2404.00456) and [Atom](https://arxiv.org/abs/2310.19102). 
