"""MLP с индикаторами пропусков."""

from torch import nn


def make_mlp(
    input_dim: int, hidden_sizes: list[int], dropout: float, activation: str = "relu"
) -> nn.Module:
    """Собирает MLP с указанными скрытыми слоями.

    Параметры
    ---------
    input_dim : int
        Размер входного признака (8 признаков + 5 индикаторов = 13).
    hidden_sizes : list[int]
        Размеры скрытых слоёв, например [64, 32].
    dropout : float
        Вероятность dropout (0.0 — выключено).
    activation : str
        'relu' | 'tanh' | 'gelu'.

    Возвращает
    ----------
    nn.Sequential — MLP с финальным линейным слоем (выход 1 логит).
    """
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    act_cls = activations[activation]

    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))

    return nn.Sequential(*layers)
