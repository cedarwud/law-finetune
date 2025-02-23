import subprocess
import time
from datetime import datetime

# 定義微調的指令
command = "llamafactory-cli train config.yaml"

# 記錄開始時間
start_time = time.time()
start_datetime = datetime.now()
print(f"微調開始時間: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# 使用 Popen 執行微調指令，實時顯示輸出
process = subprocess.Popen(
    command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # 將 stderr 合併到 stdout
    text=True
)

# 逐行讀取並顯示輸出
while True:
    line = process.stdout.readline()
    if not line and process.poll() is not None:  # 當進程結束且無輸出時退出
        break
    if line:
        print(line, end='')  # 實時顯示訓練過程

# 等待進程結束並獲取返回值
return_code = process.wait()

# 檢查是否成功
if return_code != 0:
    print(f"微調失敗！返回碼: {return_code}")
    exit(1)
else:
    print("微調執行完成！")

# 記錄結束時間
end_time = time.time()
end_datetime = datetime.now()
print(f"微調結束時間: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# 計算總耗時（單位：秒）
total_time_seconds = end_time - start_time

# 轉換為小時、分鐘、秒
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)

# 輸出結果
print(f"總耗時: {hours} 小時 {minutes} 分鐘 {seconds} 秒")
print(f"總耗時（秒）: {total_time_seconds:.2f} 秒")