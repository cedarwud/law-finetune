from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_path = "../LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-32B"
adapter_path = "../LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B-QLoRA-4bit-3280-fixed"
merged_model_path = "../LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-32B-merged"

# 🚀 使用 4-bit 量化設定，降低 VRAM 占用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 計算時使用 FP16
    bnb_4bit_use_double_quant=True,  # 開啟 double quant，減少記憶體使用
    bnb_4bit_quant_type="nf4"  # 使用 NF4 量化格式，提高效能
)

# 🚀 載入基礎模型，強制使用 `flash_attention_2`
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",  # ✅ 啟用 Flash Attention 加速推理
    device_map="auto"
)

# 🚀 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 🚀 加載 QLoRA 適配器
model = PeftModel.from_pretrained(base_model, adapter_path)

# 🚀 合併 LoRA 權重並儲存
print("開始合併 LoRA 權重...")
model = model.merge_and_unload()

# 🚀 安全儲存，避免 rounding error
model.save_pretrained(merged_model_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_path)

print("✅ 模型合併完成，已保存至:", merged_model_path)
