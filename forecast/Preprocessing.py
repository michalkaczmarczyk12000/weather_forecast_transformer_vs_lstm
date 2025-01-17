import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic


def process_data(source):
    df = pd.read_csv(source)

    timestamps = df["Date Time"]
    timestamps_hour = np.array(
        [
            float(datetime.datetime.strptime(t, "%d.%m.%Y %H:%M:%S").hour)
            for t in timestamps
        ]
    )
    timestamps_day = np.array(
        [
            float(datetime.datetime.strptime(t, "%d.%m.%Y %H:%M:%S").day)
            for t in timestamps
        ]
    )
    timestamps_month = np.array(
        [
            float(datetime.datetime.strptime(t, "%d.%m.%Y %H:%M:%S").month)
            for t in timestamps
        ]
    )

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    df["sin_hour"] = np.sin(2 * np.pi * timestamps_hour / hours_in_day)
    df["cos_hour"] = np.cos(2 * np.pi * timestamps_hour / hours_in_day)
    df["sin_day"] = np.sin(2 * np.pi * timestamps_day / days_in_month)
    df["cos_day"] = np.cos(2 * np.pi * timestamps_day / days_in_month)
    df["sin_month"] = np.sin(2 * np.pi * timestamps_month / month_in_year)
    df["cos_month"] = np.cos(2 * np.pi * timestamps_month / month_in_year)

    return df


train_dataset = process_data("Data/jena_train_raw.csv")
test_dataset = process_data("Data/jena_test_raw.csv")

train_dataset.to_csv(r"Data/jweather_train.csv", index=False)
test_dataset.to_csv(r"Data/jweather_test.csv", index=False)
