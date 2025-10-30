import torch
import numpy as np
from collections import OrderedDict
from utils.model_utils import get_model_class


class Server:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        model_name = self.config['model']['name']
        model_params = self.config['model']['config']

        ModelClass = get_model_class(model_name)
        self.global_model = ModelClass(**model_params).to(self.device)
        print(f"Server initialized with model: {model_name}")


    def aggregate_updates(self, trend_updates: list, season_updates: list):
        if not trend_updates and not season_updates:
            print("No updates received. Skipping aggregation.")
            return None, None

        print("--- Aggregating updates with standard FedAvg ---")

        agg_trend_delta, agg_season_delta = OrderedDict(), OrderedDict()

        # --- 聚合趋势组件更新 ---
        valid_trend_updates = [u for u in trend_updates if u]
        if valid_trend_updates:
            num_clients = len(valid_trend_updates)
            print(f"Aggregating trend updates from {num_clients} clients.")
            for key in valid_trend_updates[0].keys():
                # 简单平均
                aggregated_tensor = torch.stack([u[key] for u in valid_trend_updates]).mean(dim=0)
                agg_trend_delta[key] = aggregated_tensor.to(self.device)

        # --- 聚合季节组件更新 ---
        valid_season_updates = [u for u in season_updates if u]
        if valid_season_updates:
            num_clients = len(valid_season_updates)
            print(f"Aggregating season updates from {num_clients} clients.")
            for key in valid_season_updates[0].keys():
                # 简单平均
                aggregated_tensor = torch.stack([u[key] for u in valid_season_updates]).mean(dim=0)
                agg_season_delta[key] = aggregated_tensor.to(self.device)

        return agg_trend_delta, agg_season_delta

    def update_global_model(self, agg_trend_delta: OrderedDict, agg_season_delta: OrderedDict):
        """用聚合后的更新量来更新全局模型"""
        if not agg_trend_delta and not agg_season_delta:
            return
        current_state_dict = self.global_model.state_dict()
        if agg_trend_delta:
            for key, value in agg_trend_delta.items():
                if key in current_state_dict:
                    current_state_dict[key] += value
        if agg_season_delta:
            for key, value in agg_season_delta.items():
                if key in current_state_dict:
                    current_state_dict[key] += value
        self.global_model.load_state_dict(current_state_dict)
        print("Global model updated.")
