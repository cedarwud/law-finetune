import csv
import json
import os

# 取得當前腳本所在的目錄 (確保相對路徑正確)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相對路徑存取檔案
csv_file = os.path.join(base_dir, "law_3280.csv")  # 假設 CSV 檔案與程式碼同目錄
jsonl_file = os.path.join(base_dir, "law_3280.jsonl")  # 輸出 JSONL 檔案

# 設定 CSV 檔案的編碼
encoding = "big5"

# 轉換 CSV 為 JSONL
with open(csv_file, "r", encoding=encoding, errors="replace") as csv_f, open(
    jsonl_file, "w", encoding="utf-8"
) as jsonl_f:
    reader = csv.DictReader(csv_f)  # 假設第一行是標頭 (prompt, input, output)
    for row in reader:
        # 建立每筆資料的字典，確保 key 存在
        data = {
            "instruction": row.get("prompt", ""),  # 使用 get 避免 KeyError
            "input": row.get("input", ""),
            "output": row.get("output", ""),
        }
        # 寫入 JSONL 檔案，每行一個 JSON 對象
        jsonl_f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("轉換完成！JSONL 檔案已儲存至:", jsonl_file)
