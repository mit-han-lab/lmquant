# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Llama2-7B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name llama2-7b --smooth-xw-alpha 0.85 --smooth-xw-beta 0.15

# SmoothQuant (W8A8 with per-tensor static KV quantization) on Llama2-7B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/static.yaml --model-name llama2-7b --smooth-xw-alpha 0.85 --smooth-xw-beta 0.15

# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Llama2-13B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name llama2-13b --smooth-xw-alpha 0.85 --smooth-xw-beta 0.15

# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Llama2-70B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name llama2-13b --smooth-xw-alpha 0.9 --smooth-xw-beta 0.1

# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Llama3-8B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name llama3-8b --smooth-xw-alpha 0.85 --smooth-xw-beta 0.15

# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Llama3-70B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name llama3-8b --smooth-xw-alpha 0.85 --smooth-xw-beta 0.15

# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Mistral-7B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name mistral-7b --smooth-xw-alpha 0.8 --smooth-xw-beta 0.2

# SmoothQuant (W8A8 with per-token dynamic KV quantization) on Mixtral-8x7B
python -m lmquant.llm.run configs/llm.yaml configs/smoothquant/dynamic.yaml --model-name mixtral-8x7b --smooth-xw-alpha 0.8 --smooth-xw-beta 0.2

