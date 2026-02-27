import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ================= 配置区域 =================
# 1. 输入文件路径 (请确保 data.csv 就在项目根目录的 data_files 文件夹里)
# 或者你可以直接改为绝对路径，比如: r"C:\Users\Downloads\data.csv"
input_csv_path = Path("data_files/data.csv") 

# 2. 输出目录 (修改为项目内的 data/epilepsy)
output_dir = Path("data/epilepsy")
# ===========================================

# 检查输入文件是否存在
if not input_csv_path.exists():
    print(f"❌ 错误：找不到文件 {input_csv_path.resolve()}")
    print("请确认：\n1. 在项目里创建了 'data_files' 文件夹\n2. 把 'data.csv' 放进去了")
    exit()

# 自动创建输出目录
output_dir.mkdir(parents=True, exist_ok=True)

print(f"正在读取数据: {input_csv_path} ...")
data = pd.read_csv(input_csv_path)

y = data.iloc[:, -1]
x = data.iloc[:, 1:-1]

x = x.to_numpy()
y = y.to_numpy()
y = y - 1
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

for i, j in enumerate(y):
    if j != 0:
        y[i] = 1

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 保存 Train
dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, output_dir / "train.pt")

# 保存 Val
dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, output_dir / "val.pt")

# 保存 Test
dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, output_dir / "test.pt")

print(f"✅ 处理完成！文件已保存至: {output_dir.resolve()}")