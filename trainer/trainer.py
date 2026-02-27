import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

# =======================================================
    # 【在这里插入】 权重判断逻辑
    # =======================================================
    # 1. 初始化默认权重 (对应原始论文/跨视图)
    config.lambda1 = 1.0       # Cross-View 权重
    config.lambda2 = 0.7     # Context 权重
    config.lambda_self = 0  # Self-View 权重 (默认关闭)

   # 获取文件夹名称的小写形式，方便判断
    run_name = experiment_log_dir.lower()

    if "mixed" in run_name:
        # Mixed 模式
        config.lambda_self = 0.7
        print(f"👉 检测到 Mixed 模式 (lambda1={config.lambda1}, lambda_self={config.lambda_self})")
        
    elif "cross" in run_name:
        # Cross 模式 (保持默认)
        config.lambda1 = 1.0
        config.lambda_self = 0
        print(f"👉 检测到 Cross 模式 (lambda1={config.lambda1}, lambda_self={config.lambda_self})")
        
    elif "self" in run_name:
        # Self 模式 (只有当既不是mixed也不是cross，且包含self时，才进这里)
        config.lambda1 = 0 
        config.lambda_self = 1 
        print(f"👉 检测到 Self 模式 (lambda1={config.lambda1}, lambda_self={config.lambda_self})")
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # 1. 数据搬运
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # 2. 梯度清零
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        # =========================================================
        # 分支 A: 自监督预训练 (Self-Supervised) —— 【修改核心】
        # =========================================================
        if training_mode == "self_supervised":
            # 获取特征
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # 归一化
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            # -------------------------------------------------------
            # 1. 计算【跨视图】TC 损失 (Original / Cross-View)
            # -------------------------------------------------------
            # 逻辑：用 View1 的上下文预测 View2 的未来（及其反向）
            # 作用：学习对噪声和增强的不变性 (Invariance)
            tc_loss_cross1, context1 = temporal_contr_model(features1, features2)
            tc_loss_cross2, context2 = temporal_contr_model(features2, features1)
            
            # -------------------------------------------------------
            # 2. 计算【同视图】TC 损失 (New / Same-View)
            # -------------------------------------------------------
            # 逻辑：用 View1 的上下文预测 View1 自己的未来（及其反向）
            # 作用：加强对单一样本内部时序依赖的学习 (Temporal Dependency)
            # 注意：这里我们不需要返回的上下文 context，用 _ 忽略
            tc_loss_self1, _ = temporal_contr_model(features1, features1)
            tc_loss_self2, _ = temporal_contr_model(features2, features2)

            # -------------------------------------------------------
            # 3. 计算【上下文】CC 损失 (Contextual Contrasting)
            # -------------------------------------------------------
            # 使用跨视图产生的上下文向量计算一致性
            zis = context1 
            zjs = context2 
            
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss_cc = nt_xent_criterion(zis, zjs)

            # -------------------------------------------------------
            # 4. 组合总损失 (Joint Loss)
            # -------------------------------------------------------
            lambda1 = config.lambda1       # 跨视图权重 (建议保持主导)
            lambda2 = config.lambda2    # 上下文权重
            lambda_self = config.lambda_self # 【新权重】同视图权重 (建议设小一点，避免模型偷懒)

            # 总损失 = (跨视图TC) + (同视图TC) + (上下文CC)
            loss = (tc_loss_cross1 + tc_loss_cross2) * lambda1 + \
                   (tc_loss_self1 + tc_loss_self2) * lambda_self + \
                   loss_cc * lambda2

        # =========================================================
        # 分支 B: 监督/微调 (Supervised) —— 【保持原样】
        # =========================================================
        else: 
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels) # 仅计算分类损失
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        # 反向传播
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
        
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
