# ========== QoQ with Post-Scale Zero Point ===============================

# ========== QoQ (W4A8KV4 with per-channel weight quantization) ==========
# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-2-7B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-2-7b --smooth-proj-alpha 0 --smooth-proj-beta 1

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-2-13B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-2-13b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-2-70B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-2-70b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-7B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-7b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-13B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-13b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-30B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-30b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-3-8B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-3-8b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-3-70B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-3-70b --smooth-proj-alpha 0.05 --smooth-proj-beta 0.95

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-3.1-8B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-3.1-8b

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-3.1-70B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name llama-3.1-70b

# QoQ (W4A8KV4 with per-channel weight quantization) on Mistral-7B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name mistral-7b

# QoQ (W4A8KV4 with per-channel weight quantization) on Yi-34B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name yi-34b --smooth-proj-alpha 0.2 --smooth-proj-beta 0.8 --smooth-attn-strategy Manual --smooth-attn-beta 0

# QoQ (W4A8KV4 with per-channel weight quantization) on Mixtral-8x7B
python -m deepcompressor.app.llm.ptq configs/qoq-gchn.yaml --model-name mixtral-8x7b --smooth-proj-alpha 0.05 --smooth-proj-beta 0.95

# ========================================================================



# ========== QoQ (W4A8KV4 with progressive weight quantization) ==========
# QoQ (W4A8KV4 with progressive weight quantization) on Llama-2-7B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-2-7b

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-2-13B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-2-13b --smooth-proj-alpha 0.25 --smooth-proj-beta 0.75 --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-2-70B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-2-70b

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-7B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-7b

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-13B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-13b --smooth-proj-alpha 0.1 --smooth-proj-beta 0.9

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-30B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-30b --smooth-proj-alpha 0.2 --smooth-proj-beta 0.8

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3-8B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-3-8b --smooth-proj-alpha 0.35 --smooth-proj-beta 0.65 --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3-70B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-3-70b --smooth-proj-alpha 0.35 --smooth-proj-beta 0.65 --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.1-8B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-3.1-8b --smooth-proj-alpha 0.2 --smooth-proj-beta 0.8 --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.1-70B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name llama-3.1-70b --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Mistral-7B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name mistral-7b --smooth-proj-alpha 0.1 --smooth-proj-beta 0.9 --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Yi-34B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name yi-34b --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Mixtral-8x7B
python -m deepcompressor.app.llm.ptq configs/qoq-g128.yaml --model-name mixtral-8x7b --smooth-proj-alpha 0.35 --smooth-proj-beta 0.65 --smooth-attn-strategy GridSearch --smooth-attn-beta " -2"

# ========================================================================

# ========================================================================