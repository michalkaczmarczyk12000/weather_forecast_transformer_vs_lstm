import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mae(true_future, predicted_future):
    return mean_absolute_error(true_future, predicted_future)

def calculate_rmse(true_future, predicted_future):
    return np.sqrt(mean_squared_error(true_future, predicted_future))

def inverse_standardize(scaled_data, mean, std):
    return (scaled_data * std) + mean

def multi_step_plot(history, true_future, prediction, num_points=6):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_points)/STEP, np.array(true_future[:num_points]), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_points)/STEP, np.array(prediction[:num_points]), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def create_time_steps(length):
  return list(range(-length, 0))

hrs = 24
dataset = 'weatherHistory'
target = 'humidity'

if dataset == 'jenaClimate':
    csv_path = "Data\jena_climate_2009_2016.csv"
    df = pd.read_csv(csv_path)
    features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)', "VPmax (mbar)","VPact (mbar)","sh (g/kg)","H2OC (mmol/mol)", "rh (%)"]
    features = df[features_considered]
    features.index = df['Date Time']
    TRAIN_SPLIT = 300000
    if target == 'temp':
       desired_target = 1
    elif target == 'humidity':
       desired_target = -1
    else:
       print('wrong target')
    model_path = "Models/lstm_" + str(target) +"_"+ str(hrs) + "hrs_jena.keras"
    points_to_show = hrs*6
elif dataset == 'weatherHistory':
    csv_path = "Data\weatherHistory.csv"
    df = pd.read_csv(csv_path)
    df.sort_values(by='Formatted Date', inplace = True)
    features_considered = ['Temperature (C)', 'Humidity']
    features = df[features_considered]
    features.index = df['Formatted Date']
    TRAIN_SPLIT = 70000
    if target == 'temp':
       desired_target = 0
    elif target == 'humidity':
       desired_target = -1
    else:
       print('wrong target')
    model_path = "Models/lstm_" + str(target) +"_"+ str(hrs) + "hrs_weather.keras"
    points_to_show = hrs
else:
  print("Provide valid dataset")


dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

past_history = 720
STEP = 6
future_target = hrs*6

x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, desired_target],
                                            TRAIN_SPLIT, None, past_history,
                                            future_target, STEP)
BATCH_SIZE = 256
BUFFER_SIZE = 10000

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

model = load_model(model_path)

all_true_futures = []
all_predicted_futures = []

for x, y in val_data_multi.take(len(x_val_multi) // BATCH_SIZE//10):
    predictions = model.predict(x)
    
    for i in range(len(x)):
        true_future = y[i]
        predicted_future = predictions[i]

        true_future_real = inverse_standardize(true_future, data_mean[desired_target], data_std[desired_target])
        predicted_future_real = inverse_standardize(predicted_future, data_mean[desired_target], data_std[desired_target])

        all_true_futures.append(true_future_real)
        all_predicted_futures.append(predicted_future_real)

all_true_futures = np.concatenate(all_true_futures, axis=0)
all_predicted_futures = np.concatenate(all_predicted_futures, axis=0)


overall_mae = calculate_mae(all_true_futures, all_predicted_futures)
overall_rmse = calculate_rmse(all_true_futures, all_predicted_futures)

print(f'Overall MAE: {overall_mae}')
print(f'Overall RMSE: {overall_rmse}')

for x, y in val_data_multi.take(5):
  multi_step_plot(inverse_standardize(x[desired_target],data_mean[desired_target], data_std[desired_target]), 
                  inverse_standardize(y[desired_target],data_mean[desired_target], data_std[desired_target]),
                  inverse_standardize(model.predict(x)[desired_target],data_mean[desired_target], data_std[desired_target]),
                  num_points=points_to_show)