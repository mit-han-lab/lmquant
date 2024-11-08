# GPTQ-R (W4A16) on Llama2-7B
python -m deepcompressor.app.llm.ptq configs/gptq.yaml --model-name llama-2-7b

# GPTQ-R (W4A16) on Llama2-13B
python -m deepcompressor.app.llm.ptq configs/gptq.yaml --model-name llama-2-13b

# GPTQ-R (W4A16) on Llama2-70B
python -m deepcompressor.app.llm.ptq configs/gptq.yaml --model-name llama-2-70b

# GPTQ-R (W4A16) on Llama3-8B
python -m deepcompressor.app.llm.ptq configs/gptq.yaml --model-name llama-3-8b

# GPTQ-R (W4A16) on Llama3-70B
python -m deepcompressor.app.llm.ptq configs/gptq.yaml --model-name llama-3-70b