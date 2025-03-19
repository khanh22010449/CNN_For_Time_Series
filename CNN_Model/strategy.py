# """pytorch-example: A Flower / PyTorch app."""

# import json
# from logging import INFO

# import torch
# import wandb
# from CNN_Model.task import CNN, create_run_dir, set_weights

# from flwr.common import logger, parameters_to_ndarrays, Context
# from flwr.common.typing import UserConfig
# from flwr.server.strategy import FedAvg

# PROJECT_NAME = "CNN_Model for Time_series"


# class CustomFedAvg(FedAvg):
#     """A class that behaves like FedAvg but has extra functionality.

#     This strategy: (1) saves results to the filesystem, (2) saves a
#     checkpoint of the global  model when a new best is found, (3) logs
#     results to W&B if enabled.
#     """

#     def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Create a directory where to save results from this run
#         self.save_path, self.run_dir = create_run_dir(run_config)
#         self.use_wandb = use_wandb
#         # Initialise W&B if set
#         if use_wandb:
#             self._init_wandb_project()

#         # Keep track of best acc
#         self.best_r2_so_far = -float("inf")

#         # A dictionary to store results as they come
#         self.results = {}

#         # Store current parameters
#         self.current_parameters = self.initial_parameters

#     def _init_wandb_project(self):
#         # init W&B
#         wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

#     def _store_results(self, tag: str, results_dict):
#         """Store results in dictionary, then save as JSON."""
#         # Update results dict
#         if tag in self.results:
#             self.results[tag].append(results_dict)
#         else:
#             self.results[tag] = [results_dict]

#         # Save results to disk.
#         # Note we overwrite the same file with each call to this function.
#         # While this works, a more sophisticated approach is preferred
#         # in situations where the contents to be saved are larger.
#         with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
#             json.dump(self.results, fp)

#     def _update_best_model(self, server_round, r2, parameters):
#         """Update the best model if a new best is found."""
#         if r2 > self.best_r2_so_far:
#             # We have a new best model
#             self.best_r2_so_far = r2
#             # Convert parameters to ndarrays
#             params_dict = parameters_to_ndarrays(parameters)

#             # Save the model
#             model = CNN()
#             set_weights(model, params_dict)
#             torch.save(model.state_dict(), f"{self.save_path}/best_model.pth")

#             # Log
#             logger.log(
#                 INFO,
#                 "New best model found in round %s with r2=%s",
#                 server_round,
#                 r2,
#             )

#     def aggregate_fit(self, server_round, results, failures):
#         """Aggregate model weights and store the results."""
#         # Call aggregate_fit from the parent class (FedAvg)
#         parameters_aggregated, metrics_aggregated = super().aggregate_fit(
#             server_round, results, failures
#         )

#         # Store the current parameters
#         self.current_parameters = parameters_aggregated

#         # Store the fit metrics
#         if metrics_aggregated is not None:
#             metrics = {"server_round": server_round, **metrics_aggregated}
#             self._store_results("fit", metrics)
#             if self.use_wandb:
#                 wandb.log({"fit": metrics}, step=server_round)

#         return parameters_aggregated, metrics_aggregated

#     def aggregate_evaluate(self, server_round, results, failures):
#         """Aggregate evaluation metrics and store the results."""
#         # Call aggregate_evaluate from the parent class (FedAvg)
#         loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
#             server_round, results, failures
#         )

#         # Store the evaluation metrics
#         if metrics_aggregated is not None:
#             metrics = {"server_round": server_round, **metrics_aggregated}
#             self._store_results("evaluate", metrics)
#             if self.use_wandb:
#                 wandb.log({"evaluate": metrics}, step=server_round)

#             # Update the best model if we have a new best
#             self._update_best_model(
#                 server_round, metrics["r2"], self.current_parameters
#             )

#         return loss_aggregated, metrics_aggregated


"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO

import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
from CNN_Model.task import CNN, create_run_dir, set_weights

from flwr.common import logger, parameters_to_ndarrays, Context
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

PROJECT_NAME = "CNN_Model for Time_series"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_r2_so_far = -float("inf")

        # A dictionary to store results as they come
        self.results = {}

        # Lists to store MSE and R² values for plotting
        self.mse_values = []
        self.r2_values = []
        self.rounds = []

        # Store current parameters
        self.current_parameters = self.initial_parameters

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_model(self, server_round, r2, parameters):
        """Update the best model if a new best is found."""
        if r2 > self.best_r2_so_far:
            # We have a new best model
            self.best_r2_so_far = r2
            # Convert parameters to ndarrays
            params_dict = parameters_to_ndarrays(parameters)

            # Save the model
            model = CNN()
            set_weights(model, params_dict)
            torch.save(model.state_dict(), f"{self.save_path}/best_model.pth")

            # Log
            logger.log(
                INFO,
                "New best model found in round %s with r2=%s",
                server_round,
                r2,
            )

    def _plot_metrics(self):
        """Plot MSE and R² values."""
        plt.figure(figsize=(12, 5))

        # Plot MSE
        plt.subplot(1, 2, 1)
        plt.plot(self.rounds, self.mse_values, "b-o", linewidth=2)
        plt.title("MSE over Rounds")
        plt.xlabel("Round")
        plt.ylabel("MSE")
        plt.grid(True)

        # Plot R²
        plt.subplot(1, 2, 2)
        plt.plot(self.rounds, self.r2_values, "g-o", linewidth=2)
        plt.title("R² over Rounds")
        plt.xlabel("Round")
        plt.ylabel("R²")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/metrics_plot.png")

        # If using wandb, log the figure
        if self.use_wandb:
            wandb.log(
                {"metrics_plot": wandb.Image(f"{self.save_path}/metrics_plot.png")}
            )

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights and store the results."""
        # Call aggregate_fit from the parent class (FedAvg)
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Store the current parameters
        self.current_parameters = parameters_aggregated

        # Store the fit metrics
        if metrics_aggregated is not None:
            metrics = {"server_round": server_round, **metrics_aggregated}
            self._store_results("fit", metrics)
            if self.use_wandb:
                wandb.log({"fit": metrics}, step=server_round)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics and store the results."""
        # Call aggregate_evaluate from the parent class (FedAvg)
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Store the evaluation metrics
        if metrics_aggregated is not None:
            metrics = {"server_round": server_round, **metrics_aggregated}
            self._store_results("evaluate", metrics)

            # Store MSE and R² values for plotting
            if "loss" in metrics_aggregated:
                self.mse_values.append(metrics_aggregated["loss"])
                self.rounds.append(server_round)

            if "r2" in metrics_aggregated:
                self.r2_values.append(metrics_aggregated["r2"])

            # Plot metrics after each round
            if len(self.mse_values) > 0 and len(self.r2_values) > 0:
                self._plot_metrics()

            if self.use_wandb:
                wandb.log({"evaluate": metrics}, step=server_round)

            # Update the best model if we have a new best
            self._update_best_model(
                server_round, metrics["r2"], self.current_parameters
            )

        return loss_aggregated, metrics_aggregated
