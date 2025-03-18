"""pytorch-example: A Flower / PyTorch app."""

import torch

from CNN_Model.strategy import CustomFedAvg
from CNN_Model.task import CNN, get_weights, set_weights, test, load_data
from torch.utils.data import DataLoader

from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def gen_evaluate_fn(
    test_X,
    test_y,
    scaler,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = CNN()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, r2, mae = test(net, test_X, test_y, device=device, scaler=scaler)
        return loss, {"r2": r2, "mae": mae}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


def weighted_average(metrics):
    r2_values = [num_examples * m["r2"] for num_examples, m in metrics]
    mae_values = [num_examples * m["mae"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {
        "r2": sum(r2_values) / total_examples,
        "mae": sum(mae_values) / total_examples,
    }


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    # Initialize model parameters
    ndarrays = get_weights(CNN())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation

    # This is the exact same dataset as the one donwloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    _, _1, test_x, test_y, scaler = load_data()

    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(
            test_x, test_y, scaler, device=context.run_config["server-device"]
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
