import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# Класс датасета
class MazeDataset(Dataset):
    """
    Класс датасета лабиринтов
    :param mazes: лабиринты
    :param paths: пути
    """

    def __init__(self, mazes: np.ndarray, paths: np.ndarray):
        self.mazes = mazes
        self.paths = paths

    def __len__(self) -> int:
        return len(self.mazes)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        maze = self.mazes[idx]
        path = self.paths[idx]
        return maze, path


class Generator(nn.Module):
    """
    Модель генератора пути
    :param input_size: размер входа
    :param hidden_size: размер скрытого слоя
    :param output_size: размер выхода
    :param num_layers: количество слоев
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# Загрузка данных
mazes = ...  # Загрузите или создайте лабиринты
paths = ...  # Загрузите или создайте пути

# Создание датасета и даталоадера
dataset = MazeDataset(mazes, paths)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Функция обучения
def train(model: Generator,
          data_loader: DataLoader,
          loss_fn: nn.CrossEntropyLoss,
          optimizer: torch.optim.Adam):
    """
    Функция обучения модели
    :param model: модель
    :param data_loader: загрузчик данных
    :param loss_fn: функция потерь
    :param optimizer: оптимизатор
    :return: None
    """
    for maze, real_path in data_loader:
        optimizer.zero_grad()
        # Генерация пути с помощью модели
        generated_path = model(maze)

        # Вычисление функции потерь
        loss = loss_fn(generated_path, real_path)

        # Обратное распространение ошибки
        loss.backward()

        # Обновление весов модели
        optimizer.step()


# Функция потерь и оптимизатор
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

# Цикл обучения
for epoch in range(num_epochs):
    train(generator, data_loader, loss_fn, optimizer)
    # Валидация и вывод статистики
    # ...