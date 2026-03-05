import torch
import torch.nn.functional as F
import os
from pathlib import Path

# ================= 配置区域 =================
SOURCE_DIR = Path("data/pFD")          # 源数据文件夹
SUFFIXES = ['a', 'b', 'c', 'd']        # 需要处理的后缀
MODES = ['train', 'test', 'val']       # 需要处理的文件类型
ORIG_LEN = 5120                        # 原始长度

# 👇 1. 确认目标长度为 2560
TARGET_LEN = 2560                      
# ===========================================

def downsample_and_save():
    print(f"🚀 开始处理: {SOURCE_DIR} (长度 {ORIG_LEN} -> {TARGET_LEN})")
    
    if not SOURCE_DIR.exists():
        print(f"❌ 错误: 源文件夹 {SOURCE_DIR} 不存在！")
        return

    for suffix in SUFFIXES:
        # 👇 2. 【核心修改】文件夹命名格式改为: pFD_{TARGET_LEN}{suffix}
        # 结果示例: pFD_2560a, pFD_2560b
        target_dir_name = f"pFD_{TARGET_LEN}{suffix}"
        
        target_dir = Path("data") / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 正在处理分组 [{suffix}] -> 目标文件夹: {target_dir}")

        for mode in MODES:
            # 构造源文件名
            src_filename = f"{mode}_{suffix}.pt"
            src_path = SOURCE_DIR / src_filename

            if not src_path.exists():
                print(f"   ⚠️ 跳过: 找不到文件 {src_path}")
                continue

            # 加载与处理
            try:
                data = torch.load(src_path)
                
                # 解析结构
                if isinstance(data, dict):
                    samples = data['samples']
                    labels = data['labels']
                else:
                    samples = data[0]
                    labels = data[1]

                # 转 Tensor
                if not torch.is_tensor(samples):
                    samples = torch.tensor(samples)
                
                orig_shape = samples.shape
                
                # 升维方便插值
                if len(samples.shape) == 2:
                    samples = samples.unsqueeze(1)
                
                # --- 重采样 ---
                samples_down = F.interpolate(samples, size=TARGET_LEN, mode='linear', align_corners=False)
                
                # 降维
                if len(orig_shape) == 2:
                    samples_down = samples_down.squeeze(1)

                # 保存
                target_filename = f"{mode}.pt"
                target_path = target_dir / target_filename
                
                if isinstance(data, dict):
                    save_dict = data.copy()
                    save_dict['samples'] = samples_down
                    torch.save(save_dict, target_path)
                else:
                    torch.save((samples_down, labels), target_path)

                print(f"   ✅ {src_filename} -> {target_dir_name}/{target_filename}")

            except Exception as e:
                print(f"   ❌ [失败] {src_path}: {e}")

    print("\n🎉 所有数据处理完成！")

# 执行函数

downsample_and_save()
