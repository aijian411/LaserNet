import os
import glob
import pandas as pd
from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
    Dataset_Solar, Dataset_PEMS, Dataset_Pred,
    MultiSubjectPCGECG, split_files_by_subject, fit_scalers_on_train
)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'PCGECG': MultiSubjectPCGECG,
    'custom': Dataset_Custom,
}


def _detect_pcg_ecg_columns(first_file: str):
    """
    Detect whether CSV has header with 'pcg'/'ecg' column names (case-insensitive).
    If not, fallback to column indices 0 and 1.

    Returns:
        x_cols: tuple[str] or tuple[int]
        y_col: str or int
        use_header: bool
    """
    df = pd.read_csv(first_file)

    # Try case-insensitive match
    cols = list(df.columns)
    cols_lower = [str(c).lower() for c in cols]
    if "pcg" in cols_lower and "ecg" in cols_lower:
        pcg_name = cols[cols_lower.index("pcg")]
        ecg_name = cols[cols_lower.index("ecg")]
        return (pcg_name,), ecg_name, True

    # Fallback: no usable header, use first 2 columns (pcg, ecg)
    return (0,), 1, False


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    # Default loader options
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False   # IMPORTANT: do not drop test samples
        batch_size = 1       # evaluation uses bsz=1 in this repo
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    else:
        raise ValueError(f"Unknown flag={flag}")

    # ---------------- PCGECG SPECIAL BRANCH ----------------
    if args.data == 'PCGECG':
        # 1) Collect all CSV files in ONE directory
        pattern = getattr(args, "data_path", None)
        if pattern and any(ch in pattern for ch in ['*', '?', '[']):
            file_paths = sorted(glob.glob(os.path.join(args.root_path, pattern)))
        else:
            # default: all csv under root_path
            file_paths = sorted(glob.glob(os.path.join(args.root_path, "*.csv")))

        if len(file_paths) == 0:
            raise FileNotFoundError(f"No CSV files found in: {args.root_path}")

        # 2) Detect columns (header vs no header)
        x_cols, y_col, use_header = _detect_pcg_ecg_columns(file_paths[0])

        # 3) Split by subject(file) to avoid leakage (no cross-subject overlap)
        train_ratio = getattr(args, "train_ratio", 0.7)
        val_ratio = getattr(args, "val_ratio", 0.1)
        seed = getattr(args, "seed", 2021)
        train_files, val_files, test_files = split_files_by_subject(
            file_paths, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

        # 4) Fit scalers on TRAIN subjects only
        scaler_x, scaler_y = fit_scalers_on_train(train_files, x_cols=x_cols, y_col=y_col, use_header=use_header)

        # 5) Pick split files by flag
        if flag == 'train':
            split_files = train_files
        elif flag == 'val':
            split_files = val_files
        else:
            # test/pred: use test split
            split_files = test_files

        data_set = MultiSubjectPCGECG(
            file_paths=split_files,
            size=[args.seq_len, args.label_len, args.pred_len],
            x_cols=x_cols,
            y_col=y_col,
            scale=True,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            cache_files=getattr(args, "cache_files", False),
            use_header=use_header
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    # ---------------- ORIGINAL DATASETS ----------------
    Data = data_dict[args.data]
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
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
