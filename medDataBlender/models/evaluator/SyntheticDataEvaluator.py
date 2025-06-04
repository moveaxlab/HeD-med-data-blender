from abc import ABC, abstractmethod
from typing import List, Tuple
import torch


# Abstract Base Class
class SyntheticDataEvaluator(ABC):
    def __init__(
        self,
        real_data: List[Tuple[torch.Tensor, torch.Tensor]],
        synthetic_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ):

        self.real_data = real_data
        self.synthetic_data = synthetic_data

    @abstractmethod
    def evaluate(self) -> dict:
        pass
