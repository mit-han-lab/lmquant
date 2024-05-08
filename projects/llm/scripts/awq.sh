# AWQ (W4A16) on Llama2-7B
python -m lmquant.llm.run configs/llm.yaml configs/awq.yaml --model-name llama2-7b

# AWQ (W4A16) on Llama2-13B
python -m lmquant.llm.run configs/llm.yaml configs/awq.yaml --model-name llama2-13b

# AWQ (W4A16) on Llama2-70B
python -m lmquant.llm.run configs/llm.yaml configs/awq.yaml --model-name llama2-70b

# AWQ (W4A16) on Llama3-8B
python -m lmquant.llm.run configs/llm.yaml configs/awq.yaml --model-name llama3-8b

# AWQ (W4A16) on Llama3-70B
python -m lmquant.llm.run configs/llm.yaml configs/awq.yaml --model-name llama3-70b