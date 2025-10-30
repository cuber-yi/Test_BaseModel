# cuber-yi/test_basemodel/Test_BaseModel-main/client.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.model_utils import get_model_class
from utils.reporting_utils import save_client_results
import copy


class Client:
    def __init__(self, client_id: int, dataloader: DataLoader, config: dict, device: torch.device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.config = config
        self.device = device

        # --- 使用模型工厂动态创建本地模型 ---
        model_name = self.config['model']['name']
        model_params = self.config['model']['config']

        ModelClass = get_model_class(model_name)
        self.model = ModelClass(**model_params).to(self.device)

        # 用于存储从服务器接收的全局权重（只存参数）
        self.global_params = None

    def set_global_model(self, global_state_dict: OrderedDict):
        """从服务器接收并加载完整的全局模型权重"""
        self.model.load_state_dict(global_state_dict)
        # 克隆全局权重，但只克隆可训练的参数，用于后续计算更新量
        self.global_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}

    def local_train(self):
        """执行本地训练（标准FedAvg，无正则化）"""
        self.model.train()

        # 从config中获取超参数
        local_epochs = self.config['federation']['local_epochs']
        lr = self.config['training']['lr']

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(local_epochs):
            for x_batch, y_batch in self.dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch.squeeze(-1))
                loss.backward()
                optimizer.step()

    def compute_update(self):
        """
        计算模型可训练参数的更新量（delta）。
        delta = local_param - global_param
        """
        delta = OrderedDict()
        # 只遍历可训练的参数来计算更新
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta[name] = param.data - self.global_params[name]
        return delta

    def evaluate(self, save_dir: str):
        """
        在联邦学习结束后，用最终的全局模型进行评估。
        """
        print(f"\n--- Client {self.client_id}: Starting Final Evaluation ---")

        # --- 1. 保存最终模型 ---
        model_path = os.path.join(save_dir, 'models', f'client_{self.client_id}_final_model.pth')
        torch.save(self.model.state_dict(), model_path)

        # --- 2. 评估模型性能 ---
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in self.dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                all_preds.append(outputs.cpu().numpy())
                all_true.append(y_batch.squeeze(-1).cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)

        mae = mean_absolute_error(all_true, all_preds)
        rmse = np.sqrt(mean_squared_error(all_true, all_preds))

        print(f"Evaluation Metrics for Client {self.client_id}: MAE={mae:.4f}, RMSE={rmse:.4f}")

        # --- 3. 获取样本用于绘图 ---
        x_sample, y_sample = next(iter(self.dataloader))
        x_sample, y_sample = x_sample[0:1].to(self.device), y_sample[0:1]
        with torch.no_grad():
            y_pred_sample = self.model(x_sample).cpu().numpy().flatten()
        y_true_sample = y_sample.numpy().flatten()

        # --- 4. 调用外部函数保存结果和绘图 ---
        metrics = {'MAE': mae, 'RMSE': rmse}
        save_client_results(
            save_dir=save_dir,
            client_id=self.client_id,
            metrics=metrics,
            y_true=y_true_sample,
            y_pred=y_pred_sample
        )

        return mae, rmse
