def extract_time_features(dt_series, freq="ms"):
    """
    将字符串时间戳（如 1970-01-01 13:54:55.141）转换为时间特征向量
    freq: 控制提取粒度，可选：
        - 'h'：hour
        - 't'：minute, second, millisecond
        - 's'：second only
        - 'ms'：hour, minute, second, millisecond
        - 'none'：dummy zero
    """
    time_features = []
    for t in dt_series:
        try:
            dt = datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S.%f")
        except:
            dt = datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")
        row = []

        if freq == "h":
            row.append(dt.hour)

        elif freq == "t":
            row.append(dt.minute)
            row.append(dt.second)
            row.append(dt.microsecond // 1000)

        elif freq == "s":
            row.append(dt.second)

        elif freq == "ms":
            row.append(dt.hour)
            row.append(dt.minute)
            row.append(dt.second)
            row.append(dt.microsecond // 1000)

        elif freq == "none":
            row = [0]

        time_features.append(row)

    return np.array(time_features, dtype=np.float32)
