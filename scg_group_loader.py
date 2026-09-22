import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime


def extract_time_features(dt_series, freq="s"):
    """
    将字符串时间戳（如 1970-01-01 13:54:55.141）转换为时间特征向量
    freq: 控制提取粒度，可选 'h', 's', 'ms', 'none'
    """
    time_features = []
    for t in dt_series:
        try:
            dt = datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S.%f")
        except:
            dt = datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")
        row = []
        if freq in ["h", "s", "ms"]:
            row.append(dt.hour)
        if freq in ["s", "ms"]:
            row.append(dt.minute)
            row.append(dt.second)
        if freq == "ms":
            row.append(dt.microsecond // 1000)
        if freq == "none":
            row = [0]
        time_features.append(row)
    return np.array(time_features, dtype=np.float32)


class SCGGroupDataset(Dataset):
    def __init__(self, folder_path, file_list, seq_len=400, pred_len=100, freq="ms"):
        self.samples = []
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len

        for fname in file_list:
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath) or not fname.endswith(".csv"):
                continue

            df = pd.read_csv(fpath)
            if 'scg_signal' not in df.columns or 'ids_dt' not in df.columns:
                continue

            scg = df['scg_signal'].values.astype(np.float32)
            time_feat = extract_time_features(df['ids_dt'], freq=freq)

            if len(scg) < self.total_len:
                continue

            # 标准化
            scg = (scg - np.mean(scg)) / np.std(scg)

            for i in range(0, len(scg) - self.total_len):
                x = scg[i:i + seq_len]
                y = scg[i + seq_len:i + self.total_len]
                time_x = time_feat[i:i + seq_len]
                time_y = time_feat[i + seq_len:i + self.total_len]
                self.samples.append((
                    torch.tensor(x).unsqueeze(-1),       # [seq_len, 1]
                    torch.tensor(y).unsqueeze(-1),       # [pred_len, 1]
                    torch.tensor(time_x),                # [seq_len, time_dim]
                    torch.tensor(time_y)                 # [pred_len, time_dim]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
