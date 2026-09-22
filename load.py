def extract_time_features(dt_series, freq="ms"):

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
