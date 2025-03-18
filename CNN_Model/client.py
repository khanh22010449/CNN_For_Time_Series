"""pytorch: A Flower / PyTorch app."""

import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy

from CNN_Model.task import (
    CNN,
    get_weights,
    load_data,
    set_weights,
    train,
    test,
)

from flwr.common.logger import log
from logging import INFO


# Hàm chuyển đổi kiểu dữ liệu numpy sang Python native types
def convert_metrics(metrics: dict) -> dict:
    """Chuyển đổi numpy types sang Python native types"""
    converted_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            converted_metrics[k] = float(v)  # Chuyển sang Python float
        elif isinstance(v, np.ndarray):
            converted_metrics[k] = v.tolist()  # Chuyển numpy array sang list
        else:
            converted_metrics[k] = v
    return converted_metrics


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        client__state: RecordSet,
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        local_epochs,
    ):
        self.model = model
        self.client_state = client__state
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        # Set model weights
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        # Train model locally
        train_loss = train(
            self.model,
            self.X_train,
            self.y_train,
            self.local_epochs,
            self.device,
        )
        self._save_layer_weights_to_state()

        # Return updated weights, number of samples, and training loss
        return (
            get_weights(self.model),
            len(self.X_train),
            {"train_loss": float(train_loss)},  # Đảm bảo train_loss là Python float
        )

    def evaluate(self, parameters, config):
        # Set model weights
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()

        # Evaluate model on local test data
        loss, r2, mae = test(
            self.model, self.X_test, self.y_test, self.device, self.scaler
        )

        # Chuyển đổi tất cả giá trị sang Python native types
        metrics = {
            "r2": float(r2),  # Đảm bảo r2 là Python float
            "mae": float(mae),  # Đảm bảo mae là Python float
        }

        # Log để kiểm tra kiểu dữ liệu
        log(
            INFO,
            f"Metrics types - r2: {type(metrics['r2'])}, mae: {type(metrics['mae'])}",
        )

        return float(loss), len(self.X_test), metrics

    def _save_layer_weights_to_state(self):
        state_dict_arrays = {}

        for k, v in self.model.fc2.state_dict().items():
            state_dict_arrays[k] = array_from_numpy(v.cpu().numpy())

        # Add to recordset (replace if already exists)
        self.client_state.parameters_records[self.local_layer_name] = ParametersRecord(
            state_dict_arrays
        )

    def _load_layer_weights_from_state(self):

        if self.local_layer_name not in self.client_state.parameters_records:
            return

        state_dict = {}
        param_records = self.client_state.parameters_records

        for k, v in param_records[self.local_layer_name].items():
            state_dict[k] = torch.from_numpy(v.numpy())

        self.model.fc2.load_state_dict(state_dict, strict=True)


def client_fn(context: Context):
    # Load model and data
    model = CNN()
    X_train, y_train, X_test, y_test, scaler = load_data()

    # Get local epochs from context
    local_epochs = context.run_config["local-epochs"]
    client_state = context.state

    # Tạo FlowerClient
    numpy_client = FlowerClient(
        model, client_state, X_train, y_train, X_test, y_test, scaler, local_epochs
    )

    # Chuyển đổi NumPyClient sang Client để tránh cảnh báo
    return numpy_client.to_client()


# Flower ClientApp
app = ClientApp(client_fn)
