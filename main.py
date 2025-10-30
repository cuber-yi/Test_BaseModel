import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import copy
import pandas as pd
import os
import datetime
from collections import OrderedDict
from utils.config_utils import load_config
from utils.model_utils import partition_model
from utils.data_loader import load_battery_data, preprocess_data, create_windowed_dataset
from utils.reporting_utils import save_summary_report
from client import Client
from server import Server


def setup_clients_by_file(file_paths, window_size, pre_len, batch_size, max_capacity):
    """
    为每个数据文件创建一个客户端数据加载器。
    """
    client_dataloaders = []
    client_id_counter = 0
    for file_path in file_paths:
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            print(f"Loading data for Client {client_id_counter} from {file_path}. Found sheets: {sheet_names}")
            all_battery_data = load_battery_data(file_path)
            all_datasets_for_client = []
            for sheet_name in sheet_names:
                df = all_battery_data[sheet_name]
                X_scaled, y, _ = preprocess_data(df, max_capacity)
                X_windowed, y_windowed = create_windowed_dataset(X_scaled, y, window_size, pre_len)
                if len(X_windowed) == 0:
                    print(f"  - Warning: No samples created for sheet '{sheet_name}'. Skipping this sheet.")
                    continue
                X_tensor = torch.FloatTensor(X_windowed)
                y_tensor = torch.FloatTensor(y_windowed)
                dataset = TensorDataset(X_tensor, y_tensor)
                all_datasets_for_client.append(dataset)
                print(f"  - Loaded {len(dataset)} samples from sheet: {sheet_name}.")
            if all_datasets_for_client:
                combined_client_dataset = ConcatDataset(all_datasets_for_client)
                dataloader = DataLoader(combined_client_dataset, batch_size=batch_size, shuffle=True)
                client_dataloaders.append(dataloader)
                print(
                    f"--> Created DataLoader for Client {client_id_counter} with a total of {len(combined_client_dataset)} samples.")
                client_id_counter += 1
            else:
                print(f"--> Warning: No data loaded for Client from file {file_path}. This client will not be created.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
    return client_dataloaders


def setup_clients_by_sheet(file_path, window_size, pre_len, batch_size, max_capacity):
    """
    为单个文件中的每个Sheet创建一个客户端数据加载器。
    模式2: 单个文件，每个Sheet对应一个独立客户端
    """
    client_dataloaders = []
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        print(f"Loading data from single file: {file_path}")
        print(f"Found {len(sheet_names)} sheets (each will be a separate client): {sheet_names}")

        all_battery_data = load_battery_data(file_path)

        for client_id, sheet_name in enumerate(sheet_names):
            df = all_battery_data[sheet_name]
            X_scaled, y, _ = preprocess_data(df, max_capacity)
            X_windowed, y_windowed = create_windowed_dataset(X_scaled, y, window_size, pre_len)

            if len(X_windowed) == 0:
                print(f"  - Warning: No samples created for sheet '{sheet_name}'. Skipping this client.")
                continue

            X_tensor = torch.FloatTensor(X_windowed)
            y_tensor = torch.FloatTensor(y_windowed)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            client_dataloaders.append(dataloader)

            print(f"--> Created Client {client_id} from sheet '{sheet_name}' with {len(dataset)} samples.")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

    return client_dataloaders


def main():
    # --- 加载配置 ---
    config = load_config('config/config.yaml')
    device = torch.device(config['data']['device'])
    print(f"Configuration loaded for model '{config['model']['name']}'. Running on device: {device}")

    # --- 创建结果保存目录 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config['results']['save_dir_prefix']}{timestamp}"
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    print(f"Results will be saved in: {save_dir}")

    # --- 准备数据和客户端 ---
    data_mode = config['data'].get('mode', 'multi_file')

    if data_mode == 'single_file_multi_client':
        # 单个文件，每个Sheet一个客户端
        print(f"\n{'=' * 50}")
        print("Data Loading Mode: Single File, Multiple Clients (one client per sheet)")
        print(f"{'=' * 50}\n")

        single_file_path = config['data'].get('single_file', None)
        if not single_file_path:
            print("Error: 'single_file' not specified in config for 'single_file_multi_client' mode.")
            return

        client_dataloaders = setup_clients_by_sheet(
            file_path=single_file_path,
            window_size=config['data']['window_size'],
            pre_len=config['data']['pre_len'],
            batch_size=config['federation']['batch_size'],
            max_capacity=config['data']['max_capacity']
        )
    else:
        # 多个文件，每个文件一个客户端
        print(f"\n{'=' * 50}")
        print("Data Loading Mode: Multiple Files (one client per file)")
        print(f"{'=' * 50}\n")

        client_dataloaders = setup_clients_by_file(
            file_paths=config['data']['files'],
            window_size=config['data']['window_size'],
            pre_len=config['data']['pre_len'],
            batch_size=config['federation']['batch_size'],
            max_capacity=config['data']['max_capacity']
        )
    if not client_dataloaders:
        print("No data loaded. Exiting.")
        return

    num_total_clients = len(client_dataloaders)
    print(f"\nTotal number of clients created: {num_total_clients}")

    # 初始化Client和Server实例，传入完整的config对象
    clients = [Client(client_id=i, dataloader=dl, config=config, device=device) for
               i, dl in enumerate(client_dataloaders)]
    server = Server(config=config, device=device)

    # --- 联邦学习主循环 ---
    num_rounds = config['federation']['num_rounds']
    for comm_round in range(num_rounds):
        print(f"\n{'=' * 20} Communication Round {comm_round + 1}/{num_rounds} {'=' * 20}")

        # --- 模型划分与分发 ---
        partitions = partition_model(server.global_model, config['model']['partition'])
        global_trend_state = partitions.get('trend', {}).get('state_dict', OrderedDict())
        global_season_state = partitions.get('season', {}).get('state_dict', OrderedDict())

        client_trend_updates, client_season_updates = [], []

        # --- 客户端本地训练与更新计算 ---
        for client in clients:
            client.set_global_model(copy.deepcopy(global_trend_state), copy.deepcopy(global_season_state))
            client.local_train()
            delta_trend, delta_season = client.compute_update()
            client_trend_updates.append(delta_trend)
            client_season_updates.append(delta_season)

        # --- 服务器端聚合 ---
        agg_trend_delta, agg_season_delta = server.aggregate_updates(
            client_trend_updates, client_season_updates
        )
        # 更新全局模型
        server.update_global_model(agg_trend_delta, agg_season_delta)

    print("\nFederated learning process finished.")

    # --- 个性化训练与评估 ---
    print(f"\n{'=' * 20} Starting Personalization Phase {'=' * 20}")
    # 获取最终的全局模型并分发
    final_partitions = partition_model(server.global_model, config['model']['partition'])
    final_global_trend = final_partitions.get('trend', {}).get('state_dict', OrderedDict())
    final_global_season = final_partitions.get('season', {}).get('state_dict', OrderedDict())

    for client in clients:
        client.set_global_model(copy.deepcopy(final_global_trend), copy.deepcopy(final_global_season))

    all_metrics = []
    for client in clients:
        mae, rmse = client.personalize_and_evaluate(save_dir=save_dir)
        all_metrics.append({'client_id': client.client_id, 'MAE': mae, 'RMSE': rmse})

    # --- 6. 打印最终结果摘要 ---
    print(f"\n{'=' * 20} Final Evaluation Summary {'=' * 20}")
    avg_mae = np.mean([m['MAE'] for m in all_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in all_metrics])
    for metrics in all_metrics:
        print(f"Client {metrics['client_id']}: MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    print(f"\nAverage across all clients: MAE = {avg_mae:.4f}, RMSE = {avg_rmse:.4f}")

    # --- 7. 保存摘要 ---
    avg_metrics = {'MAE': avg_mae, 'RMSE': avg_rmse}
    save_summary_report(save_dir=save_dir, all_metrics=all_metrics, avg_metrics=avg_metrics)


if __name__ == '__main__':
    main()
