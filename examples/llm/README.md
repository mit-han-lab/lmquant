# QServe: *W4A8KV4* Quantization for Efficient LLM Serving

[[Website](https://hanlab.mit.edu/projects/qserve)][[Paper](https://arxiv.org/abs/2405.04532)][[QServe GPU Inference System](https://github.com/mit-han-lab/qserve)]

Quantization can accelerate large language model (LLM) inference. Going beyond INT8 quantization, the research community is actively exploring even lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization techniques only accelerate low-batch, edge LLM inference, failing to deliver performance gains in large-batch, cloud-based LLM serving. We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when **dequantizing either weights or partial sums** on GPUs. To address this challenge, we introduce **QoQ**, a W4A8KV4 quantization algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands for **quattuor-octo-quattuor**, which represents 4-8-4 in Latin. QoQ is implemented by the **QServe** inference library that achieves measured speedup. The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by **operations on low-throughput CUDA cores**. Building upon this insight, in QoQ algorithm, we introduce progressive quantization that can allow low dequantization overhead in W4A8 GEMM. Additionally, we develop SmoothAttention to effectively mitigate the accuracy degradation incurred by 4-bit KV quantization. In the QServe system, we perform compute-aware weight reordering and take advantage of register-level parallelism to reduce dequantization latency. We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization. As a result, QServe improves the maximum achievable serving throughput of Llama-3-8B by **1.2×** on A100, **1.4×** on L40S; and Qwen1.5-72B by **2.4×** on A100, **3.5×** on L40S, compared to TensorRT-LLM.

![QoQ-QServe](/assets/llm/qoq/qoq-qserve.png)
![QoQ](/assets/llm/qoq/qoq.png)

## Usage

The following command will perform per-channel QoQ quantization (W4A8KV4) and evaluate the quantized model on Wikitext-2:
```bash
python -m deepcompressor.app.llm.ptq \
    configs/qoq-gchn.yaml \
    --model-name llama-2-7b --model-path /PATH/TO/LLAMA-2-7B \
    --smooth-proj-alpha 0 --smooth-proj-beta 1 \
    --smooth-attn-alpha 0.5 --smooth-attn-beta 0
```

In this command,
- The positional arguments are configuration files which are loaded in order. [`configs/qoq-gchn.yaml`](configs/qoq-gchn.yaml) contains the quantization configurations specialized in QoQ per-channel W4A8KV4 quantization. Please make sure all configuration files are under a subfolder of the working directory where you run the command.
- All configurations can be directly set in either YAML file or command line. Please refer to [`configs/__default__.yaml`](configs/llm.yaml) and `python -m deepcompressor.app.llm.ptq -h`.
- `--model-name llama-2-7b` specifies the model name, e.g., llama-30b, llama-3-8b, mixtral-8x7b.
- `--model-path /PATH/TO/LLAMA-2-7B` specifies the path to the llama-2-7b model directory. If your model directories are organized as `PATH_TO_ROOT_DIR/MODEL_FAMILY/MODEL_NAME` (e.g., `~/models/llama-2/llama-2-7b`), you can simply specify `--model-root PATH_TO_ROOT_DIR` (e.g., ```--model-root ~/models```).
- `--smooth-proj-alpha 0` specifies the alpha for SmoothLinear to be 0. `--smooth-attn-alpha 0.5` specifies the alpha for SmoothAttention to be 0.5.
- The default task is [GPTQ-style](https://github.com/IST-DASLab/gptq/blob/main/llama.py#L218) Wikitext2 perplexity evaluation. If you would like to evaluate the accuracy on zero-shot tasks such as Hellaswag using [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness), please add `--eval-tasks EVAL_TASK [EVAL_TASK ...] --eval-evaluators lm_eval` to the command. You can use `--eval-tasks zero-shot --eval-evaluators lm_eval` which will automatically add [wikitext, hellaswag, piqa, winogrande, arc_easy, arc_challenge](/deepcompressor/llm/eval.py#L51) to the evaluation tasks.
- If you would like to save quantized model checkpoint, please add `--save-model true` in the command.

## Deployment

### Deployment with Qserve Engine

If you save the QoQ W4A8KV4 quantized model checkpoint, you can easily to deploy quantized model with [`QServe`](https://github.com/mit-han-lab/qserve) engine.

Please run the following command to convert the saved checkpoint to QServe-compatible checkpoint:
```bash
python -m deepcompressor.backend.qserve.convert \
    --model-path /PATH/TO/HUGGINGCE-MODEL \
    --quant-path /PATH/TO/QUANTIZED-MODEL \
    --weight-bits 4 \
    --group-size GROUP_SIZE \
    --output-root /ROOT/PATH/TO/OUTPUT-MODEL/DIRECTORY
```

After we have the QServe-compatible checkpoint, please switch to QServe conda environment, run [qserve_e2e_generation.py](https://github.com/mit-han-lab/qserve/tree/main/qserve_e2e_generation.py) to deploy quantized model with QServe Engine.

```bash
conda deactivate
conda activate qserve
cd /PATH/TO/QSERVE
python qserve_e2e_generation.py \
  --model /PATH/TO/OUTPUT-MODEL \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path /PATH/TO/OUTPUT-MODEL \
  --group-size GROUP_SIZE
```

Please refer to [`QServe`](https://github.com/mit-han-lab/qserve) for further details.

### Deployment with TinyChat Engine

If you save the 4-bit weight quantized model checkpoint by running the following command,

```bash
python -m deepcompressor.app.llm.ptq \
    configs/awq.yaml \
    --model-name llama-3-8b-instruct --model-path /PATH/TO/LLAMA-3-8B-INSTRUCT
```
you can easily to deploy quantized model with [`TinyChat`](https://github.com/mit-han-lab/llm-awq) engine.

Please run the following command to convert the saved checkpoint to TinyChat-compatible checkpoint:
```bash
python -m deepcompressor.backend.tinychat.convert \
    --model-name MODEL_NAME \
    --quant-path /PATH/TO/QUANTIZED-MODEL \
    --group-size GROUP_SIZE \
    --output-root /ROOT/PATH/TO/OUTPUT-MODEL/DIRECTORY
```

After we have the TinyChat-compatible checkpoint, please switch to TinyChat conda environment, run [demo.py](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat/demo.py) to deploy quantized model with TinyChat Engine.

```bash
conda deactivate
conda activate tinychat
cd /PATH/TO/TINYCHAT
python demo.py --model_type llama \
    --model-path /PATH/TO/LLAMA-3-8B-INSTRUCT \
    --q_group_size GROUP_SIZE \
    --load_quant /PATH/TO/OUTPUT-MODEL \ 
    --precision W4A16
```

Please refer to [`TinyChat`](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) for further details.

## Evaluation Resutls

### Perplexity Evaluation

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

### Efficiency Benchmarks

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