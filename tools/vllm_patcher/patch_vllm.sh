# copy to source tree
cp tools/vllm_patcher/afmoe.py vllm/model_executor/models/afmoe.py

# patch into registry
sed -i 's/_TEXT_GENERATION_MODELS = {/_TEXT_GENERATION_MODELS = {"AfmoeForCausalLM": ("afmoe", "AfmoeForCausalLM"),/' vllm/model_executor/models/registry.py