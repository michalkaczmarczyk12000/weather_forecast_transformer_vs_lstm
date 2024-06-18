import torch
import torch.nn as nn
import numpy as np
from joblib import load
from DataLoader import WeatherDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from model import Transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

def calculate_metrics(predictions, targets):
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return mae, rmse

def inference_and_metrics(
    path_to_model,
    model_name,
    test_dataloader,
    device,
    scaler_path="scalar_item.joblib",
):
    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(f"{path_to_model}/{model_name}"))
    model.eval()

    criterion = nn.MSELoss()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for index_in, index_tar, _input, target in test_dataloader:
            src = _input.permute(1, 0, 2).double().to(device)[1:, :, :]
            target = target.permute(1, 0, 2).double().to(device)

            next_input_model = src
            predictions = []

            for i in range(target.shape[0] - 1):
                prediction = model(next_input_model)

                if i == 0:
                    predictions = prediction
                else:
                    predictions = torch.cat(
                        (predictions, prediction[-1, :, :].unsqueeze(0))
                    )

                pos_encoding_old_vals = src[i + 1:, :, 1:]
                pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1)
                pos_encodings = torch.cat(
                    (pos_encoding_old_vals, pos_encoding_new_val)
                )

                next_input_model = torch.cat(
                    (
                        src[i + 1:, :, 0].unsqueeze(-1),
                        prediction[-1, :, :].unsqueeze(0),
                    )
                )
                next_input_model = torch.cat(
                    (next_input_model, pos_encodings), dim=2
                )

            true_values = torch.cat((src[1:, :, 0], target[:-1, :, 0]))
            all_targets.append(true_values.cpu().numpy())
            all_predictions.append(predictions[:, :, 0].cpu().numpy())

    mae, rmse = calculate_metrics(all_predictions, all_targets)
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    
    return mae, rmse

def main(
    test_csv,
    path_to_model,
    model_name,
    device="cpu",
    training_length=48,
    forecast_window=6,
):
    test_dataset = WeatherDataset(
        csv_name=test_csv,
        root_dir="Data/",
        training_length=training_length,
        forecast_window=forecast_window,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mae, rmse = inference_and_metrics(
        path_to_model, model_name, test_dataloader, device
    )
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    test_csv = "jweather_test.csv"
    path_to_model = "save_model"
    model_name = "best_train_788.pth"
    device = "cpu"

    main(test_csv, path_to_model, model_name, device)
