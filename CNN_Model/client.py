"""pytorch: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
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
        local_epochs,
    ):
        self.model = model
        self.client_state = client__state
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
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
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        # Set model weights
        set_weights(self.model, parameters)
        self.load_layer_weights_from_state()

        # Evaluate model on local test data
        loss = test(self.model, self.X_test, self.y_test, self.device)

        # Return loss, number of test samples, and an empty dictionary (no accuracy for regression)
        return loss, len(self.X_test), {}

    def _save_layer_weights_to_state(self):
        state_dict_arrays = {}

        for k, v in self.model.fc.state_dict().items():
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
    X_train, y_train, X_test, y_test = load_data()

    # Get local epochs from context
    local_epochs = context.run_config["local-epochs"]
    client_state = context.state

    # Return the FlowerClient instance
    return FlowerClient(
        model, client_state, X_train, y_train, X_test, y_test, local_epochs
    )


# Flower ClientApp
app = ClientApp(client_fn)
