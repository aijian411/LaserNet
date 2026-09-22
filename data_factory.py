from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, Dataset_Pred
from data_provider.scg_group_loader import SCGGroupDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    if args.data == "scg_multi":
        folder = args.root_path
        file_list = [f for f in os.listdir(folder) if f.endswith('.csv')]

        # 仅划分 train/val
        train_files, val_files = train_test_split(file_list, train_size=0.8, random_state=42)

        if flag == 'train':
            used_files = train_files
        elif flag == 'val':
            used_files = val_files
        else:
            raise ValueError(f"Flag '{flag}' not supported. Only 'train' and 'val' are available for scg_multi.")

        dataset = SCGGroupDataset(
            folder_path=folder,
            file_list=used_files,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            freq=args.freq
        )

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(flag == 'train'),
            num_workers=args.num_workers,
            drop_last=True
        )

        return dataset, data_loader

    # 其他原有数据集分支
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
