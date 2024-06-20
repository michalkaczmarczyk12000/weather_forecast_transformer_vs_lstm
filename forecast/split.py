import pandas as pd

file_path = "Data/lst_dataset.csv"
data = pd.read_csv(file_path)


full_hour_data = data.iloc[5::6, :]
full_hour_data = full_hour_data.head(10000)

split_index = int(0.8 * len(full_hour_data))


train_data = full_hour_data[:split_index]
test_data = full_hour_data[split_index:]


train_data.to_csv("Data/jena_train_raw.csv", index=False)
test_data.to_csv("Data/jena_test_raw.csv", index=False)
