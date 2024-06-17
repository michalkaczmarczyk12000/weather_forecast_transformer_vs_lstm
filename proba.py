# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset without attempting to parse dates
df = pd.read_csv('Data\jena_climate_2009_2016.csv')

# Assume 'temperature' is the column we want to forecast
# Ensure your CSV has a 'temperature' column or adjust this to the correct column name
data = df[['T (degC)']].values  # Use .values to ensure 'data' is a NumPy array, required for the next steps

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM model
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length)]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Prepare the data
sequence_length = 30
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

# Predicting and inverse transforming the predictions
predicted_temperature = model.predict(X_test)
predicted_temperature = scaler.inverse_transform(predicted_temperature)

# Inverse transform the actual temperature for comparison
actual_temperature = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualization
plt.figure(figsize=(10,6))
plt.plot(actual_temperature, color='blue', label='Actual Temperature')
plt.plot(predicted_temperature, color='red', linestyle='--', label='Predicted Temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()