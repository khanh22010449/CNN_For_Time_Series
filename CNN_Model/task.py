import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
import json
from pathlib import Path
from flwr.common.typing import UserConfig


"""
class CNN(nn.Module):
    def __init__(self, input_channels=1, num_features=6, output_size=1):
        super(CNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Calculate the size after the convolutional layers
        # After two Conv1D layers with padding=1, the output size remains the same as the input size
        # num_features is the length of the input sequence
        self.flattened_size = (
            32 * num_features
        )  # This assumes no pooling layers are used

        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, output_size)  # Output layer

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debugging line
        x = F.relu(self.conv1(x))  # First convolutional layer
        x = F.relu(self.conv2(x))  # Second convolutional layer
        x = x.view(x.size(0), -1)  # Flatten tensor (batch_size, 32 * num_features)
        print(f"Flattened shape: {x.shape}")  # Debugging line
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        print(f"Output shape: {x.shape}")  # Debugging line
        return x.squeeze()  # Remove any extra dimensions


# Example of how to create an instance of this model
# model = CNN(input_channels=1, num_features=6, output_size=1)


def load_data():
    # Load your data here (this is just an example)
    X, y = make_regression(n_samples=1000, n_features=6, random_state=32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Reshape the input data to have shape (batch_size, input_channels, num_features)
    train_X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (num_samples, 1, num_features)
    test_X = torch.tensor(X_test, dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (num_samples, 1, num_features)
    train_y = torch.tensor(
        y_train, dtype=torch.float32
    ).squeeze()  # Shape: (num_samples,)
    test_y = torch.tensor(
        y_test, dtype=torch.float32
    ).squeeze()  # Shape: (num_samples,)

    return train_X, train_y, test_X, test_y
"""


class CNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=50, output_size=1):
        super(CNN, self).__init__()

        # Các lớp tích chập 1D cho dữ liệu chuỗi thời gian
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Tính toán kích thước đầu ra sau các lớp tích chập và pooling
        # Giả sử không có thay đổi kích thước qua các lớp conv (do padding=1)
        # và pooling giảm kích thước đi một nửa mỗi lần
        feature_size = sequence_length // 4  # Sau 2 lớp maxpool

        # Các lớp fully connected
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def load_data():
    # 1. Tạo hoặc tải dữ liệu chuỗi thời gian
    np.random.seed(42)
    n_samples = 1000
    time_steps = 50  # Độ dài chuỗi thời gian (window size)

    # Tạo chuỗi thời gian với xu hướng, chu kỳ và nhiễu
    time = np.arange(n_samples + time_steps)
    # Thêm xu hướng
    trend = 0.01 * time
    # Thêm chu kỳ
    seasonality = 0.5 * np.sin(2 * np.pi * time / 24)  # Chu kỳ 24 đơn vị thời gian
    # Thêm nhiễu
    noise = 0.1 * np.random.randn(len(time))

    # Tổng hợp chuỗi thời gian
    time_series = trend + seasonality + noise

    # 2. Chuẩn hóa dữ liệu (quan trọng cho CNN)
    mean = np.mean(time_series)
    std = np.std(time_series)
    time_series_normalized = (time_series - mean) / std

    # 3. Chuẩn bị dữ liệu dạng cửa sổ trượt (sliding window)
    X = []
    y = []

    for i in range(n_samples):
        # X: cửa sổ time_steps điểm dữ liệu
        # y: giá trị tiếp theo cần dự đoán
        X.append(time_series_normalized[i : i + time_steps])
        y.append(time_series_normalized[i + time_steps])

    X = np.array(X)
    y = np.array(y)

    # 4. Chia tập huấn luyện/kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 5. Định dạng dữ liệu cho CNN1D
    # CNN1D cần đầu vào dạng [batch_size, channels, sequence_length]
    train_X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    test_X = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    train_y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    test_y = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # 6. Lưu lại mean và std để inverse transform khi cần
    scaler = {"mean": mean, "std": std}

    return train_X, train_y, test_X, test_y, scaler


def train(net, X_train, y_train, epochs: int, device, batch_size=64):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # Chuyển sang mini-batch để tránh overfitting
    dataset_size = len(X_train)
    losses = []

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle data mỗi epoch
        indices = torch.randperm(dataset_size)

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]

            batch_X = X_train[batch_indices].to(device)
            batch_y = y_train[batch_indices].to(device)

            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_indices)

        avg_epoch_loss = epoch_loss / dataset_size
        losses.append(avg_epoch_loss)
        scheduler.step(avg_epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

    return losses[-1]  # Trả về loss của epoch cuối cùng


def test(net, X_test, y_test, device, scaler=None):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    net.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Move data to the appropriate device
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # Forward pass
        predictions = net(X_test)

        # Calculate loss
        loss = criterion(predictions, y_test)

        # Convert to numpy for metrics calculation
        y_test_np = y_test.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # Inverse transform nếu có scaler
        if scaler:
            y_test_np = y_test_np * scaler["std"] + scaler["mean"]
            predictions_np = predictions_np * scaler["std"] + scaler["mean"]

        # Tính các metrics
        r2 = r2_score(y_test_np, predictions_np)
        mae = mean_absolute_error(y_test_np, predictions_np)

    print(f"Test Loss: {loss.item():.6f}, R²: {r2:.6f}, MAE: {mae:.6f}")
    return loss.item(), r2, mae


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_run_dir(config=None):
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    if config:
        with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
            json.dump(config, fp)

    return save_path, run_dir


if __name__ == "__main__":
    # Thiết lập seed cho tính tái tạo
    torch.manual_seed(42)
    np.random.seed(42)

    # Tải dữ liệu
    X_train, y_train, X_test, y_test, scaler = load_data()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Khởi tạo mô hình
    sequence_length = X_train.shape[2]  # Lấy độ dài chuỗi từ dữ liệu
    net = CNN(input_channels=1, sequence_length=sequence_length, output_size=1)

    # Thiết lập device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Huấn luyện mô hình
    epochs = 100
    avg_trainloss = train(net, X_train, y_train, epochs, device)
    print(f"Final training loss: {avg_trainloss:.6f}")

    # Đánh giá mô hình
    loss, r2, mae = test(net, X_test, y_test, device, scaler)

    # Hiển thị kết quả
    print(f"Test Results - Loss: {loss:.6f}, R²: {r2:.6f}, MAE: {mae:.6f}")
