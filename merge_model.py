import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

base_model_path = "../LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-32B"
adapter_path = "../LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B-QLoRA-4bit-3280-fixed"
merged_model_path = "../LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-32B-merged"

# 加載基礎模型到 CPU
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu",  # 強制使用 CPU，避免 GPU 溢出
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 檢查是否有嵌套並解包
if hasattr(base_model, 'model'):
    base_model = base_model.model

# 加載適配器
print("加載適配器...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 清理記憶體
del base_model
gc.collect()

# 在 CPU 上合併 LoRA 權重
print("開始合併 LoRA 權重...")
model = model.merge_and_unload()

# 分片保存模型
print("正在保存模型...")
model.save_pretrained(
    merged_model_path,
    safe_serialization=True,
    max_shard_size="1GB"  # 每個分片 1GB，減少記憶體需求
)
tokenizer.save_pretrained(merged_model_path)

print("✅ 模型合併完成，已保存至:", merged_model_path)
