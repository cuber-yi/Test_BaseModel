import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.model_utils import get_model_class, partition_model
from utils.reporting_utils import save_client_results


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

        # 用于存储从服务器接收的全局权重，以及本地参数列表
        self.global_trend_weights = None
        self.global_season_weights = None
        self.local_trend_params = None
        self.local_season_params = None

    def set_global_model(self, global_trend_state_dict: OrderedDict, global_season_state_dict: OrderedDict):
        """从服务器接收并加载全局模型权重，并根据配置划分本地模型参数"""
        # 加载所有参与联邦的参数（非个性化层）
        self.model.load_state_dict(global_trend_state_dict, strict=False)
        self.model.load_state_dict(global_season_state_dict, strict=False)

        # 使用通用划分函数来获取本地模型的参数列表
        partitions = partition_model(self.model, self.config['model']['partition'])

        self.local_trend_params = partitions.get('trend', {}).get('params', [])
        self.local_season_params = partitions.get('season', {}).get('params', [])

        # 克隆对应的全局权重，用于计算本地训练时的正则化损失和最终的更新量
        self.global_trend_weights = [p.clone().detach() for p in self.local_trend_params]
        self.global_season_weights = [p.clone().detach() for p in self.local_season_params]

    def local_train(self):
        """执行本地训练，所有超参数均从config中读取"""
        self.model.train()

        # 从config中获取超参数
        local_epochs = self.config['federation']['local_epochs']
        lr = self.config['training']['lr']
        lambda_trend = self.config['regularization']['lambda_trend']
        lambda_season = self.config['regularization']['lambda_season']

        # 优化器现在训练模型的全部参数，包括个性化层
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(local_epochs):
            for x_batch, y_batch in self.dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(x_batch)
                prediction_loss = criterion(outputs, y_batch.squeeze(-1))

                # 正则化项依然只针对联邦部分，防止其偏离全局模型太远
                trend_reg_loss = 0.0
                if self.local_trend_params and self.global_trend_weights:
                    for local_p, global_p in zip(self.local_trend_params, self.global_trend_weights):
                        trend_reg_loss += torch.norm(local_p - global_p, p=2) ** 2

                season_reg_loss = 0.0
                if self.local_season_params and self.global_season_weights:
                    for local_p, global_p in zip(self.local_season_params, self.global_season_weights):
                        season_reg_loss += torch.norm(local_p - global_p, p=2) ** 2

                total_loss = prediction_loss + (lambda_trend / 2) * trend_reg_loss + \
                             (lambda_season / 2) * season_reg_loss

                total_loss.backward()
                optimizer.step()

    def compute_update(self):
        """
        计算模型更新量（delta）。
        """
        delta_trend = OrderedDict()
        delta_season = OrderedDict()

        # 再次调用划分函数，以获取训练后更新的参数名称和值
        partitions = partition_model(self.model, self.config['model']['partition'])
        trend_info = partitions.get('trend', {})
        season_info = partitions.get('season', {})

        # 计算趋势组件的更新
        if trend_info and self.global_trend_weights:
            current_state_dict = self.model.state_dict()
            for i, name in enumerate(trend_info.get('names', [])):
                param = current_state_dict[name]
                update = param.data - self.global_trend_weights[i]
                # 移除了噪声添加
                delta_trend[name] = update

        # 计算季节组件的更新
        if season_info and self.global_season_weights:
            current_state_dict = self.model.state_dict()
            for i, name in enumerate(season_info.get('names', [])):
                param = current_state_dict[name]
                update = param.data - self.global_season_weights[i]
                # 移除了噪声添加
                delta_season[name] = update

        return delta_trend, delta_season

    def personalize_and_evaluate(self, save_dir: str):
        """
        在联邦学习结束后，解冻并训练个性化层，然后进行评估。
        """
        print(f"\n--- Client {self.client_id}: Starting Personalization and Evaluation ---")

        # 1. 寻找个性化层参数
        partitions = partition_model(self.model, self.config['model']['partition'])
        federated_param_names = set(partitions.get('trend', {}).get('names', [])) | \
                                set(partitions.get('season', {}).get('names', []))

        personal_params = [
            param for name, param in self.model.named_parameters()
            if name not in federated_param_names and param.requires_grad
        ]

        if not personal_params:
            print(f"Client {self.client_id}: No personalization layers found. Evaluating the global model directly.")
        else:
            # --- 2. 训练个性化层 ---
            self.model.train()
            personalization_epochs = self.config['personalization']['epochs']
            lr = self.config['personalization']['lr']
            optimizer = torch.optim.Adam(personal_params, lr=lr)
            criterion = nn.MSELoss()
            print(f"Training personalization layer(s) for {personalization_epochs} epochs...")
            for epoch in range(personalization_epochs):
                for x_batch, y_batch in self.dataloader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(x_batch)
                    loss = criterion(outputs, y_batch.squeeze(-1))
                    loss.backward()
                    optimizer.step()

        # --- 3. 保存个性化模型 ---
        model_path = os.path.join(save_dir, 'models', f'client_{self.client_id}_personalized_model.pth')
        torch.save(self.model.state_dict(), model_path)

        # --- 4. 评估模型性能 ---
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

        # --- 5. 获取样本用于绘图 ---
        x_sample, y_sample = next(iter(self.dataloader))
        x_sample, y_sample = x_sample[0:1].to(self.device), y_sample[0:1]
        with torch.no_grad():
            y_pred_sample = self.model(x_sample).cpu().numpy().flatten()
        y_true_sample = y_sample.numpy().flatten()

        # --- 6. 调用外部函数保存结果和绘图 ---
        metrics = {'MAE': mae, 'RMSE': rmse}
        save_client_results(
            save_dir=save_dir,
            client_id=self.client_id,
            metrics=metrics,
            y_true=y_true_sample,
            y_pred=y_pred_sample
        )

        return mae, rmse
