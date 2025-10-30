import torch
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

    def aggregate_updates(self, client_updates: list):
        """
        使用FedAvg聚合所有客户端的完整模型更新。
        """
        if not client_updates:
            print("No updates received. Skipping aggregation.")
            return None

        print(f"--- Aggregating updates from {len(client_updates)} clients with standard FedAvg ---")

        # 初始化一个空的聚合delta
        agg_delta = OrderedDict()

        # 获取第一个客户端更新的所有键
        sample_update = client_updates[0]
        keys = sample_update.keys()

        for key in keys:
            # 收集所有客户端对当前参数的更新，并计算平均值
            aggregated_tensor = torch.stack([update[key] for update in client_updates]).mean(dim=0)
            agg_delta[key] = aggregated_tensor.to(self.device)

        return agg_delta

    def update_global_model(self, agg_delta: OrderedDict):
        """用聚合后的更新量来更新全局模型"""
        if not agg_delta:
            return

        current_state_dict = self.global_model.state_dict()
        for key, value in agg_delta.items():
            if key in current_state_dict:
                current_state_dict[key] += value  # 应用更新: new_weight = old_weight + delta

        self.global_model.load_state_dict(current_state_dict)
        print("Global model updated.")
