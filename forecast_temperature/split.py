import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytaj dane z pliku CSV
file_path = 'Data\weatherHistory.csv'
data = pd.read_csv(file_path)

# Podział na zbiór treningowy i testowy w proporcji 80:20
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Zapisz wyniki do plików CSV
train_data.to_csv('Data/weather_train_raw.csv', index=False)
test_data.to_csv('Data/weather_test_raw.csv', index=False)