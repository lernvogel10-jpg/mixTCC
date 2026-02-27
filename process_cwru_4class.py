import scipy.io
import torch
import numpy as np
import os
import shutil
from pathlib import Path

def process_cwru_4class():
    # ================= 🔧 4分类核心配置 =================
    
    # 1. 根目录
    CWRU_ROOT = Path(r"D:\Mycode\HAR_Project\CWRU")
    
    # 2. 输出目录 (建议用新名字，以免覆盖10分类数据)
    OUTPUT_DIR = Path(r"D:\Mycode\HAR_Project\data\cwru_4class")
    
    # 3. 样本长度
    SEQ_LEN = 1024
    
    # 4. 路径前缀
    DIR_12K = "12k Drive End Bearing Fault Data"
    DIR_NORMAL = "Normal Baseline"
    
    # 5. 类别映射 (4分类)
    # 格式: Label: [ (子文件夹, [文件名列表]), (子文件夹, [文件名列表]), ... ]
    DATA_MAP = {
        # --- Class 0: Normal (健康) ---
        0: [
            (DIR_NORMAL, ["normal_0.mat", "normal_1.mat", "normal_2.mat", "normal_3.mat"])
        ],
        
        # --- Class 1: Ball Fault (滚动体故障 - 合并 007, 014, 021) ---
        1: [
            (os.path.join(DIR_12K, r"Ball\0007"), ["B007_0.mat", "B007_1.mat", "B007_2.mat", "B007_3.mat"]),
            (os.path.join(DIR_12K, r"Ball\0014"), ["B014_0.mat", "B014_1.mat", "B014_2.mat", "B014_3.mat"]),
            (os.path.join(DIR_12K, r"Ball\0021"), ["B021_0.mat", "B021_1.mat", "B021_2.mat", "B021_3.mat"])
        ],
        
        # --- Class 2: Inner Race Fault (内圈故障 - 合并 007, 014, 021) ---
        2: [
            (os.path.join(DIR_12K, r"Inner Race\0007"), ["IR007_0.mat", "IR007_1.mat", "IR007_2.mat", "IR007_3.mat"]),
            (os.path.join(DIR_12K, r"Inner Race\0014"), ["IR014_0.mat", "IR014_1.mat", "IR014_2.mat", "IR014_3.mat"]),
            (os.path.join(DIR_12K, r"Inner Race\0021"), ["IR021_0.mat", "IR021_1.mat", "IR021_2.mat", "IR021_3.mat"])
        ],
        
        # --- Class 3: Outer Race Fault (外圈故障 - 合并 007, 014, 021 @6:00) ---
        3: [
            (os.path.join(DIR_12K, r"Outer Race\Centered\0007"), ["OR007@6_0.mat", "OR007@6_1.mat", "OR007@6_2.mat", "OR007@6_3.mat"]),
            (os.path.join(DIR_12K, r"Outer Race\Centered\0014"), ["OR014@6_0.mat", "OR014@6_1.mat", "OR014@6_2.mat", "OR014@6_3.mat"]),
            (os.path.join(DIR_12K, r"Outer Race\Centered\0021"), ["OR021@6_0.mat", "OR021@6_1.mat", "OR021@6_2.mat", "OR021@6_3.mat"])
        ]
    }
    # ========================================================

    print(f"{'='*60}")
    print(f"🚀 启动 CWRU 4分类 (故障部位识别) 数据处理")
    print(f"📂 根目录: {CWRU_ROOT}")
    print(f"📂 输出至: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_X, all_Y = [], []
    
    # 遍历 4 大类
    for label, folder_groups in DATA_MAP.items():
        print(f"🔹 处理 Class {label} ...")
        
        # 遍历该大类下的所有子文件夹 (例如 Ball 下的 007, 014, 021)
        for rel_path, files in folder_groups:
            folder_path = CWRU_ROOT / rel_path
            
            if not folder_path.exists():
                print(f"   ⚠️ 文件夹不存在 (跳过): {rel_path}")
                continue
                
            for fname in files:
                fpath = folder_path / fname
                if not fpath.exists(): continue
                
                try:
                    mat = scipy.io.loadmat(str(fpath))
                    # 自动寻找 DE_time
                    keys = [k for k in mat.keys() if 'DE_time' in k]
                    if not keys: continue
                    
                    raw_data = mat[keys[0]].flatten()
                    
                    # 归一化
                    mean_val = np.mean(raw_data)
                    std_val = np.std(raw_data)
                    if std_val == 0: std_val = 1
                    norm_data = (raw_data - mean_val) / std_val
                    
                    # 切分
                    n_samples = len(norm_data) // SEQ_LEN
                    if n_samples < 1: continue
                    
                    samples = norm_data[:n_samples*SEQ_LEN].reshape(n_samples, SEQ_LEN, 1)
                    labels_arr = np.full(n_samples, label, dtype=int)
                    
                    all_X.append(torch.tensor(samples, dtype=torch.float32))
                    all_Y.append(torch.tensor(labels_arr, dtype=torch.long))
                    
                    print(f"   ✅ {fname} -> Class {label}")
                    
                except Exception as e:
                    print(f"   ❌ Error {fname}: {e}")

    if not all_X:
        print("\n❌ 未提取到数据！")
        return

    full_X = torch.cat(all_X, dim=0)
    full_Y = torch.cat(all_Y, dim=0)
    
    # 打印统计
    unique, counts = np.unique(full_Y.numpy(), return_counts=True)
    print(f"\n📊 4分类 统计结果:")
    labels_name = {0:"Normal", 1:"Ball", 2:"Inner", 3:"Outer"}
    for u, c in zip(unique, counts):
        print(f"   Class {u} ({labels_name.get(u,'Unknown')}): {c} 样本")
    
    # 划分保存
    indices = torch.randperm(len(full_Y))
    n_train = int(len(full_Y) * 0.8)
    n_test = int(len(full_Y) * 0.1)
    
    splits = {
        "train.pt": indices[:n_train],
        "test.pt":  indices[n_train:n_train+n_test],
        "val.pt":   indices[n_train+n_test:]
    }
    
    for name, idx in splits.items():
        torch.save({"samples": full_X[idx], "labels": full_Y[idx]}, OUTPUT_DIR / name)
        
    print(f"\n🎉 4分类数据已就绪: {OUTPUT_DIR}")
    print(f"   (请记得使用 cwru_4class_Configs.py)")

if __name__ == "__main__":
    process_cwru_4class()