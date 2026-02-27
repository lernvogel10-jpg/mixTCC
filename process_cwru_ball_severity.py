import scipy.io
import torch
import numpy as np
import os
import shutil
from pathlib import Path

def process_cwru_ball_severity():
    # ================= 🔧 基础配置 =================
    CWRU_ROOT = Path(r"D:\Mycode\HAR_Project\CWRU\12k Drive End Bearing Fault Data")
    OUTPUT_DIR = Path(r"D:\Mycode\HAR_Project\data\cwru_ball_severity")
    SEQ_LEN = 1024
    
    # 类别映射
    DATA_MAP = {
        0: (r"Ball\0007", ["B007_0.mat", "B007_1.mat", "B007_2.mat", "B007_3.mat"]),
        1: (r"Ball\0014", ["B014_0.mat", "B014_1.mat", "B014_2.mat", "B014_3.mat"]),
        2: (r"Ball\0021", ["B021_0.mat", "B021_1.mat", "B021_2.mat", "B021_3.mat"]),
    }
    # ===============================================

    print(f"{'='*60}")
    print(f"🚀 CWRU 数据重构: 50%训练池 | 25%测试 | 25%验证")
    print(f"{'='*60}\n")
    
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_X, all_Y = [], []
    
    # --- 1. 读取数据 ---
    for label, (sub_folder, files) in DATA_MAP.items():
        folder_path = CWRU_ROOT / sub_folder
        if not folder_path.exists(): continue
        for fname in files:
            fpath = folder_path / fname
            if not fpath.exists(): continue
            try:
                mat = scipy.io.loadmat(str(fpath))
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
                samples = norm_data[:n_samples*SEQ_LEN].reshape(n_samples, SEQ_LEN, 1)
                labels_arr = np.full(n_samples, label, dtype=int)
                
                all_X.append(torch.tensor(samples, dtype=torch.float32))
                all_Y.append(torch.tensor(labels_arr, dtype=torch.long))
            except Exception as e:
                print(f"   Error {fname}: {e}")

    full_X = torch.cat(all_X, dim=0)
    full_Y = torch.cat(all_Y, dim=0)
    total_samples = len(full_Y)
    print(f"📊 总样本数: {total_samples}")

    # --- 2. 按比例切分 (50% / 25% / 25%) ---
    # 训练池用于后续生成 1%, 2%... 50% 的子集
    n_train_pool = int(total_samples * 0.50) 
    n_test = int(total_samples * 0.25)
    # 剩下的给验证集
    
    torch.manual_seed(42) # 固定种子，保证测试集永远一致
    indices = torch.randperm(total_samples)
    
    idx_train = indices[:n_train_pool]
    idx_test  = indices[n_train_pool : n_train_pool + n_test]
    idx_val   = indices[n_train_pool + n_test :]
    
    print(f"✂️  划分详情:")
    print(f"   Train Pool (最大可用): {len(idx_train)} (约50%)")
    print(f"   Test Set   (固定):     {len(idx_test)} (约25%)")
    print(f"   Val Set    (固定):     {len(idx_val)} (约25%)")
    
    splits = {
        "train.pt": idx_train, # 这是大池子，之后Step1会从中再切分
        "test.pt":  idx_test,
        "val.pt":   idx_val
    }
    
    for name, idx in splits.items():
        torch.save({"samples": full_X[idx], "labels": full_Y[idx]}, OUTPUT_DIR / name)
        
    print(f"\n🎉 数据重置完成")

if __name__ == "__main__":
    process_cwru_ball_severity()