import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytaj dane z pliku CSV
file_path = 'Data\weatherHistory.csv'
df = pd.read_csv(file_path)
df.sort_values(by='Formatted Date', inplace = True)

# Załóżmy, że mamy DataFrame df
train_size = int(0.8 * len(df))  # 80% na trening
train_data = df[:train_size]
test_data = df[train_size:]

# Zapisz wyniki do plików CSV
train_data.to_csv('Data/weather_train_raw.csv', index=False)
test_data.to_csv('Data/weather_test_raw.csv', index=False)