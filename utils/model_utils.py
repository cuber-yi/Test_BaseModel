from collections import OrderedDict
import torch.nn as nn
from src.xPatch import xPatch
from src.RNN import RNN



# 2. 创建一个模型注册表字典
MODEL_REGISTRY = {
    'xpatch': xPatch,
    'rnn': RNN,
}


def get_model_class(name: str):
    """
    根据模型名称字符串从注册表中获取模型类。
    这是一个模型工厂函数。
    """
    model_class = MODEL_REGISTRY.get(name.lower())
    if model_class is None:
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return model_class


# --- 模型参数划分工具 ---
def partition_model(model: nn.Module, partition_config: dict):
    """
    根据配置规则，将模型的参数划分为 'trend', 'season' 等部分。
    'personal' 分区被认为是客户端本地的，不会包含在返回结果中。

    参数:
        model (torch.nn.Module): 待划分的模型。
        partition_config (dict): 包含 'name_maps' 和 'default_partition' 的配置字典。
                                 通常是 config['model']['partition']。

    返回:
        一个字典，包含每个联邦分区（如 'trend', 'season'）的
        参数列表('params')、状态字典('state_dict')和参数名称列表('names')。
    """
    name_maps = partition_config.get('name_maps', {})
    default_partition = partition_config.get('default_partition', 'season')

    # 将匹配规则按前缀长度降序排序，确保优先匹配更具体（更长）的前缀
    # 例如，规则 'net.fc8' 会在 'net.' 之前被检查
    sorted_maps = sorted(name_maps.items(), key=lambda item: len(item[0]), reverse=True)

    # 初始化用于存储联邦参数的分区
    partitions = {
        'trend': {'params': [], 'state_dict': OrderedDict(), 'names': []},
        'season': {'params': [], 'state_dict': OrderedDict(), 'names': []}
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        assigned_partition = None
        # 查找匹配的分区规则
        for prefix, partition_name in sorted_maps:
            if name.startswith(prefix):
                assigned_partition = partition_name
                break

        # 如果没有找到匹配项，使用默认分区
        if assigned_partition is None:
            assigned_partition = default_partition

        # 如果分区是 'personal'，则跳过，不参与任何联邦过程
        if assigned_partition == 'personal':
            continue

        # 将参数信息添加到对应的联邦分区中
        if assigned_partition in partitions:
            partitions[assigned_partition]['params'].append(param)
            partitions[assigned_partition]['state_dict'][name] = param.data.clone()
            partitions[assigned_partition]['names'].append(name)

    return partitions
