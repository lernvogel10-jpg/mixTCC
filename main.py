import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--load_path', type=str, default=None, help='Path to the pretrained model checkpoint')
args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

# ================= [修改] 强制加载权重逻辑 =================
    # 针对迁移学习：检查当前实验目录下是否有预先放入的 ckp_last.pt
chk_path = os.path.join(experiment_log_dir, 'saved_models', 'ckp_last.pt')
    
    # 仅在非自监督模式下尝试加载（避免预训练时覆盖自己）
if training_mode != "self_supervised":
        if os.path.exists(chk_path):
            logger.debug(f"🔥 [Transfer Learning] Found checkpoint: {chk_path}")
            logger.debug("Loading pre-trained weights...")
            
            checkpoint = torch.load(chk_path)
            
            # 兼容处理：检查是否有 'model_state_dict' 键
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint.keys():
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # 加载参数 (strict=False 允许分类头不匹配，这是迁移学习的关键)
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.debug("✅ Pre-trained weights loaded successfully!")
            except Exception as e:
                logger.debug(f"⚠️ Warning during loading weights: {e}")
        else:
            logger.debug(f"No checkpoint found at {chk_path}. Training from scratch (Random Init).")
    # ================= [修改结束] =================


if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.


# ======================================================
# 🔧 [科研编程] 优化器策略修改：微调模式降权保护
#    (将这段代码替换掉原来的 model_optimizer = ... )
# ======================================================

if training_mode == "fine_tune":
    # 策略：微调时，主干网络非常脆弱，必须用极小的学习率呵护
    # 通常建议为原学习率的 1/10 甚至 1/100
    ft_lr = configs.lr * 1  # 建议先试 0.01 (降低100倍) 或者 0.1 (降低10倍)
    configs.num_epoch = 100
    
    print(f"\n🛡️ [Strategy] 激活微调保护机制：")
    print(f"   - 原始学习率: {configs.lr}")
    print(f"   - 微调学习率: {ft_lr}")
    print(f"   - 目的: 防止随机初始化的分类头破坏预训练特征\n")
    
    model_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=ft_lr, 
        betas=(configs.beta1, configs.beta2), 
        weight_decay=3e-4
    )
else:
    # 监督学习(蓝线) 或 自监督预训练，保持火力全开
    model_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=configs.lr, 
        betas=(configs.beta1, configs.beta2), 
        weight_decay=3e-4
    )
# ======================================================


temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# =========================================================
# 📢 [新版] 强行加载指定路径的模型
# =========================================================
if args.load_path:
    load_path = args.load_path
    print(f"\n🔍 [指令] 正在加载指定模型: {load_path}")

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 过滤掉分类头 (logits)，只加载特征提取器
        state_dict = {k: v for k, v in state_dict.items() if 'logits' not in k}

        # 加载
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ [成功] 预训练权重已加载！(Missing keys: {len(missing)})")
        logger.debug(f"Loaded weights from {load_path}")
    else:
        print(f"❌ [错误] 指定的路径不存在: {load_path}")
        # 这里如果不退出，就会变成随机初始化，一定要注意
# =========================================================

# Trainer(...) # 原有代码
    
# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

if training_mode != "self_supervised":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")
