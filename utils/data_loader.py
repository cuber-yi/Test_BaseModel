import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler


def load_battery_data(file_path):
    """
    加载Excel文件中的电池数据，每个Sheet对应一个电池。
    返回字典，键为Sheet名称，值为DataFrame。
    """
    xls = pd.ExcelFile(file_path)
    battery_data = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        battery_data[sheet_name] = df
    return battery_data


def preprocess_data(df, max_capacity):
    all_columns = df.columns.tolist()
    features = [col for col in all_columns if col != 'label']

    X = df[features].values
    y = df['label'].values / max_capacity

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def create_windowed_dataset(X, y, window_size, pre_len):
    """
    对单个电池的数据进行窗口化处理，特征包括物理特征和过去window_size个SOH值。
    参数：
        X: 物理特征矩阵，形状 (时间步, 物理特征数量)
        y: SOH数组，形状 (时间步,)
        window_size: 窗口大小
        pre_len: 预测的未来SOH长度
    返回：
        X_windowed: 窗口化后的特征，形状 (样本数, window_size, 物理特征数量 + 1)
        y_windowed: 对应的未来pre_len个SOH标签，形状 (样本数, pre_len)
    """
    num_samples = len(X) - window_size - pre_len + 1
    X_windowed = []
    y_windowed = []

    for i in range(num_samples):
        # 提取窗口的物理特征
        X_window = X[i:i + window_size]
        # 提取窗口的SOH值作为附加特征
        soh_window = y[i:i + window_size].reshape(-1, 1)
        # 合并特征：(window_size, N_features) + (window_size, 1) -> (window_size, N_features + 1)
        X_window_combined = np.concatenate([X_window, soh_window], axis=1)
        X_windowed.append(X_window_combined)
        # 提取未来pre_len个SOH值作为标签
        y_windowed.append(y[i + window_size:i + window_size + pre_len])

    X_windowed = np.array(X_windowed)
    y_windowed = np.array(y_windowed)

    # 仅保留有效样本（确保y_windowed的长度为pre_len）
    valid_indices = [i for i in range(len(y_windowed)) if len(y_windowed[i]) == pre_len]
    X_windowed = X_windowed[valid_indices]
    y_windowed = y_windowed[valid_indices]

    return X_windowed, y_windowed


def generate_datasets(file_sheet_map, window_size=50, pre_len=5, batch_size=32, max_capacity=2.0):
    """
    为训练集、验证集和测试集生成DataLoader。
    参数：
        file_sheet_map: 字典，键为数据集类型（'pretrain', 'finetune', 'test'），值为列表[(file_path, sheet_name), ...]
        window_size: 窗口大小
        pre_len: 预测步数
        batch_size: 批次大小
        max_capacity: 最大容量
    返回：
        dataloaders: 字典，键为'pretrain', 'finetune', 'test'，值为对应的DataLoader
        scalers: 字典，键为(file_path, sheet_name)，值为对应的MinMaxScaler
    """
    dataloaders = {'pretrain': [], 'finetune': [], 'test': []}
    scalers = {}

    for dataset_type in ['pretrain', 'finetune', 'test']:
        datasets = []
        for file_path, sheet_name in file_sheet_map.get(dataset_type, []):
            # 加载数据
            battery_data = load_battery_data(file_path)
            if sheet_name not in battery_data:
                print(f"Warning: Sheet {sheet_name} not found in {file_path}")
                continue

            # 预处理数据
            X_scaled, y, scaler = preprocess_data(battery_data[sheet_name], max_capacity)
            scalers[(file_path, sheet_name)] = scaler

            # 窗口化数据
            X_windowed, y_windowed = create_windowed_dataset(X_scaled, y, window_size, pre_len)

            # 转换为PyTorch Tensor
            X_tensor = torch.FloatTensor(X_windowed)
            y_tensor = torch.FloatTensor(y_windowed)

            # 创建TensorDataset
            dataset = TensorDataset(X_tensor, y_tensor)
            datasets.append(dataset)

        # 合并数据集并创建DataLoader
        if datasets:
            combined_dataset = ConcatDataset(datasets)
            dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=(dataset_type == 'pretrain'))
            dataloaders[dataset_type] = dataloader
        else:
            dataloaders[dataset_type] = None
            print(f"No data for {dataset_type} set")

    return dataloaders, scalers



if __name__ == "__main__":

    EXCEL_FILE_PATH = "C:/Users/31480/Desktop/小论文二/v4/data/batch-1.xlsx"

    SHEET_FOR_PRETRAIN = 'Sheet1'
    SHEET_FOR_TEST = 'Sheet2'

    # --- 超参数设置 ---
    MAX_CAP = 2.0  # 电池最大容量
    WINDOW_SIZE = 50  # 窗口大小
    PRE_LEN = 5  # 预测步数
    BATCH_SIZE = 32  # 批次大小

    # 1. 定义数据划分
    file_sheet_map = {
        'pretrain': [(EXCEL_FILE_PATH, SHEET_FOR_PRETRAIN)],
        'finetune': [(EXCEL_FILE_PATH, SHEET_FOR_PRETRAIN)],
        'test': [(EXCEL_FILE_PATH, SHEET_FOR_TEST)]
    }

    dataloaders, scalers = generate_datasets(
        file_sheet_map,
        window_size=WINDOW_SIZE,
        pre_len=PRE_LEN,
        batch_size=BATCH_SIZE,
        max_capacity=MAX_CAP,
    )

    print("\n--- 数据集加载结果 ---")

    # 检查预训练集 DataLoader
    pretrain_loader = dataloaders.get('pretrain')
    if pretrain_loader:
        # 尝试从 DataLoader 中取出一个批次的数据进行形状检查
        X_batch, y_batch = next(iter(pretrain_loader))

        # 获取特征总数（物理特征数 + 1个SOH特征）
        total_features = X_batch.shape[2]

        print(f"预训练集 '{SHEET_FOR_PRETRAIN}' 数据量: {len(pretrain_loader.dataset)} 个样本")
        print(f"特征数量 (物理特征+SOH): {total_features}")
        print(f"Batch X 形状: {X_batch.shape}")
        print(f"Batch y 形状: {y_batch.shape}")
    else:
        print(f"预训练集 {SHEET_FOR_PRETRAIN} 未成功加载。")

    print("-" * 20)

    # 检查测试集 DataLoader
    test_loader = dataloaders.get('test')
    if test_loader:
        X_batch, y_batch = next(iter(test_loader))

        # 获取特征总数（物理特征数 + 1个SOH特征）
        total_features_test = X_batch.shape[2]

        print(f"测试集 '{SHEET_FOR_TEST}' 数据量: {len(test_loader.dataset)} 个样本")
        print(f"特征数量 (物理特征+SOH): {total_features_test}")
        print(f"Batch X 形状: {X_batch.shape}")
        print(f"Batch y 形状: {y_batch.shape}")
    else:
        print(f"测试集 {SHEET_FOR_TEST} 未成功加载。")

