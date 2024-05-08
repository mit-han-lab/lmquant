# Large Foundation Model Quantization (LMQuant)

LMQuant is an open source large foundation models quantization toolbox based on PyTorch. 
LMQuant is implemented by [QServe](https://github.com/mit-han-lab/qserve), an efficient GPU inference library.

The current release supports:
- SmoothQuant, AWQ, GPTQ-R, and QoQ quantization for large language models


## News
- [2024/05] 🔥 Our latest **W4A8KV4** LLM quantization work **QoQ** algorithm and **QServe** system is publicly released! **QoQ** is short for *quattuor-octō-quattuor* which is 4-8-4 in latin. Check our [paper](https://arxiv.org/abs/2405.04532)!

## Contents
- [Large Models Quantization (LMQuant)](#large-models-quantization-lmquant)
  - [News](#news)
  - [Installation](#installation)
  - [Highlights](#highlights)
    - [QServe: W4A8KV4 Quantization for Efficient LLM Serving](#qserve-w4a8kv4-quantization-for-efficient-llm-serving)
      - [Model Zoo](#model-zoo)
      - [Perplexity Evaluation](#perplexity-evaluation)
      - [Efficiency Benchmarks](#efficiency-benchmarks)
  - [Support List](#support-list)
  - [Reference](#reference)
  - [Related Projects](#related-projects)
  - [Acknowledgement](#acknowledgement)

## Installation

1. Clone this repository and navigate to lmquant folder
```
git clone https://github.com/mit-han-lab/lmquant
cd lmquant
```

2. Install Package
```
conda env create -f environment.yml
poetry install
```

## Highlights

### QServe: W4A8KV4 Quantization for Efficient LLM Serving

[[Website](https://hanlab.mit.edu/projects/qserve)][[Paper](https://arxiv.org/abs/2405.04532)][[QoQ Algorithm Code](/projects/llm)][[QServe GPU System](https://github.com/mit-han-lab/qserve)]

Quantization can accelerate large language model (LLM) inference. Going beyond INT8 quantization, the research community is actively exploring even lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization techniques only accelerate low-batch, edge LLM inference, failing to deliver performance gains in large-batch, cloud-based LLM serving. We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when **dequantizing either weights or partial sums** on GPUs. To address this challenge, we introduce **QoQ**, a W4A8KV4 quantization algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands for **quattuor-octo-quattuor**, which represents 4-8-4 in Latin. QoQ is implemented by the **QServe** inference library that achieves measured speedup. The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by **operations on low-throughput CUDA cores**. Building upon this insight, in QoQ algorithm, we introduce progressive quantization that can allow low dequantization overhead in W4A8 GEMM. Additionally, we develop SmoothAttention to effectively mitigate the accuracy degradation incurred by 4-bit KV quantization. In the QServe system, we perform compute-aware weight reordering and take advantage of register-level parallelism to reduce dequantization latency. We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization. As a result, QServe improves the maximum achievable serving throughput of Llama-3-8B by **1.2×** on A100, **1.4×** on L40S; and Qwen1.5-72B by **2.4×** on A100, **3.5×** on L40S, compared to TensorRT-LLM.

![QoQ-QServe](/assets/llm/qoq/qoq-qserve.png)
![QoQ](/assets/llm/qoq/qoq.png)

#### Model Zoo

We provide QoQ quantized model checkpoints in [`QServe`](https://github.com/mit-han-lab/qserve) for your reference.


#### Perplexity Evaluation

Below is the WikiText2 perplexity evaluated with 2048 sequence length. The lower is the better.

| Models      | Precision    | Llama-3 8B | Llama-2 7B | Llama-2 13B | Llama-2 70B | Llama 7B | Llama 13B | Llama 30B | Mistral 7B | Yi 34B |
|-------------|--------------|------------|------------|-------------|-------------|----------|-----------|-----------|------------|--------|
| FP16        |              | 6.14       | 5.47       | 4.88        | 3.32        | 5.68     | 5.09      | 4.10      | 5.25       | 4.60   |
| SmoothQuant | W8A8         | 6.28       | 5.54       | 4.95        | 3.36        | 5.73     | 5.13      | 4.23      | 5.29       | 4.69   |
| GPTQ-R      | W4A16 g128   | 6.56       | 5.63       | 4.99        | 3.43        | 5.83     | 5.20      | 4.22      | 5.39       | 4.68   |
| AWQ         | W4A16 g128   | 6.54       | 5.60       | 4.97        | 3.41        | 5.78     | 5.19      | 4.21      | 5.37       | 4.67   |
| QuaRot      | W4A4         | 8.33       | 6.19       | 5.45        | 3.83        | 6.34     | 5.58      | 4.64      | 5.77       | NaN    |
| Atom        | W4A4 g128    | 7.76       | 6.12       | 5.31        | 3.73        | 6.25     | 5.52      | 4.61      | 5.76       | 4.97   |
| QoQ         | W4A8KV4      | 6.89       | 5.75       | 5.12        | 3.52        | 5.93     | 5.28      | 4.34      | 5.45       | 4.74   |
| QoQ         | W4A8KV4 g128 | 6.76       | 5.70       | 5.08        | 3.47        | 5.89     | 5.25      | 4.28      | 5.42       | 4.76   |

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

## Support List

### Large Language Model Quantization
| Models  | Sizes       | [QoQ (W4A8KV4)](/projects/llm/scripts/qoq.sh) | [AWQ (W4A16)](/projects/llm/scripts/awq.sh) | [GPTQ-R (W4A16)](/projects/llm/scripts/gptq.sh) | [SmoothQuant (W8A8)](/projects/llm/scripts/smoothquant.sh) |
| ------- | ------------| ------------- | ----------- | ---------------| ------------------ |
| Llama3  | 8B/70B      | ✅             | ✅          | ✅             | ✅                 |
| Llama2  | 7B/13B/70B  | ✅             | ✅          | ✅             | ✅                 |
| Llama   | 7B/13B/30B  | ✅             | ✅          | ✅             | ✅                 |
| Mistral | 7B          | ✅             | ✅          | ✅             | ✅                 |
| Mixtral | 8x7B        | ✅             | ✅          | ✅             | ✅                 |
| Yi      | 34B         | ✅             | ✅          | ✅             | ✅                 |

## Reference

If you find `lmquant` useful or relevant to your research, please kindly cite our paper:

```
@article{lin2024qserve,
  title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
  author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
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

- [**Algorithm**] [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://github.com/mit-han-lab/streaming-llm)

- [**Hardware**] [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning](https://arxiv.org/abs/2012.09852)


## Acknowledgement

LMQuant is inspired by many open-source libraries, including (but not limited to) [GPTQ](https://arxiv.org/abs/2210.17323), [QuaRot](https://arxiv.org/abs/2404.00456) and [Atom](https://arxiv.org/abs/2310.19102). 