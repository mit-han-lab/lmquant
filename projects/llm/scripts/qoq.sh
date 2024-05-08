
# ========== QoQ (W4A8KV4 with per-channel weight quantization) ==========
# QoQ (W4A8KV4 with per-channel weight quantization) on Llama2-7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama2-7b --smooth-xw-alpha 0 --smooth-xw-beta 1

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama2-13B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama2-13b --smooth-xw-alpha 0.15 --smooth-xw-beta 0.85

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama2-70B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama2-70b --smooth-xw-alpha 0.15 --smooth-xw-beta 0.85

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama-7b --smooth-xw-alpha 0.05 --smooth-xw-beta 0.95

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-13B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama-13b --smooth-xw-alpha 0.1 --smooth-xw-beta 0.9 --smooth-yx-strategy GridSearch --smooth-yx-beta " -2"

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama-30B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama-30b --smooth-xw-alpha 0.1 --smooth-xw-beta 0.9

# QoQ (W4A8KV4 with per-channel weight quantization) on Llama3-8B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name llama3-8b --smooth-xw-alpha 0.05 --smooth-xw-beta 0.95 --smooth-yx-strategy GridSearch --smooth-yx-beta " -2"

# QoQ (W4A8KV4 with per-channel weight quantization) on Mistral-7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name mistral-7b --smooth-xw-alpha 0.1 --smooth-xw-beta 0.9 --smooth-yx-strategy GridSearch --smooth-yx-beta " -2"

# QoQ (W4A8KV4 with per-channel weight quantization) on Yi-34B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name yi-34b --smooth-xw-alpha 0.2 --smooth-xw-beta 0.8

# QoQ (W4A8KV4 with per-channel weight quantization) on Mixtral-8x7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/gchn.yaml --model-name mixtral-8x7b --smooth-xw-alpha 0.05 --smooth-xw-beta 0.95

# ========================================================================



# ========== QoQ (W4A8KV4 with progressive weight quantization) ==========
# QoQ (W4A8KV4 with progressive weight quantization) on Llama2-7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama2-7b --smooth-xw-alpha 0.3 --smooth-xw-beta 0.7

# QoQ (W4A8KV4 with progressive weight quantization) on Llama2-13B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama2-13b --smooth-xw-alpha 0.25 --smooth-xw-beta 0.75

# QoQ (W4A8KV4 with progressive weight quantization) on Llama2-70B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama2-70b --smooth-xw-alpha 0.3 --smooth-xw-beta 0.7

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama-7b --smooth-xw-alpha 0.3 --smooth-xw-beta 0.7

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-13B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama-13b --smooth-xw-alpha 0.1 --smooth-xw-beta 0.9 --smooth-yx-strategy GridSearch --smooth-yx-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-30B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama-30b --smooth-xw-alpha 0.2 --smooth-xw-beta 0.8

# QoQ (W4A8KV4 with progressive weight quantization) on Llama3-8B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name llama3-8b --smooth-xw-alpha 0.3 --smooth-xw-beta 0.7 --smooth-yx-strategy GridSearch --smooth-yx-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Mistral-7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name mistral-7b --smooth-xw-alpha 0.15 --smooth-xw-beta 0.85 --smooth-yx-strategy GridSearch --smooth-yx-beta " -2"

# QoQ (W4A8KV4 with progressive weight quantization) on Yi-34B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name yi-34b --smooth-xw-alpha 0.3 --smooth-xw-beta 0.7

# QoQ (W4A8KV4 with progressive weight quantization) on Mixtral-8x7B
python -m lmquant.llm.run configs/llm.yaml configs/qoq/g128.yaml --model-name mixtral-8x7b --smooth-xw-alpha 0.2 --smooth-xw-beta 0.8

# ========================================================================