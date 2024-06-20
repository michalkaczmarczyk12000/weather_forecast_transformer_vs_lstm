from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import WeatherDataset
import logging
import time  # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def inference(
    path_to_save_predictions,
    forecast_window,
    dataloader,
    device,
    path_to_save_model,
    best_model,
):

    device = torch.device(device)

    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model + best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():

        model.eval()
        for plot in range(25):

            for index_in, index_tar, _input, target in dataloader:

                src = _input.permute(1, 0, 2).double().to(device)[1:, :, :]
                target = target.permute(1, 0, 2).double().to(device)

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window - 1):

                    prediction = model(next_input_model)

                    if all_predictions == []:
                        all_predictions = prediction
                    else:
                        all_predictions = torch.cat(
                            (all_predictions, prediction[-1, :, :].unsqueeze(0))
                        )

                    pos_encoding_old_vals = src[i + 1 :, :, 1:]
                    pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1)
                    pos_encodings = torch.cat(
                        (pos_encoding_old_vals, pos_encoding_new_val)
                    )

                    next_input_model = torch.cat(
                        (
                            src[i + 1 :, :, 0].unsqueeze(-1),
                            prediction[-1, :, :].unsqueeze(0),
                        )
                    )
                    next_input_model = torch.cat(
                        (next_input_model, pos_encodings), dim=2
                    )

                true = torch.cat((src[1:, :, 0], target[:-1, :, 0]))
                loss = criterion(true, all_predictions[:, :, 0])
                val_loss += loss

            val_loss = val_loss / 10
            scaler = load("scalar_item.joblib")
            src_humidity = scaler.inverse_transform(src[:, :, 0].cpu())
            target_humidity = scaler.inverse_transform(target[:, :, 0].cpu())
            prediction_humidity = scaler.inverse_transform(
                all_predictions[:, :, 0].detach().cpu().numpy()
            )
            plot_prediction(
                plot,
                path_to_save_predictions,
                src_humidity,
                target_humidity,
                prediction_humidity,
                index_in,
                index_tar,
            )

        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")
