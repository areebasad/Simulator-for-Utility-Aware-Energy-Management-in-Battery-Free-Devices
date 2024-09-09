from pathlib import Path
from typing import Any, Dict, List
from .component import Component
import pandas as pd
import yaml


from abc import abstractmethod


class Device(Component):
    """Base class for a device."""

    def __init__(
        self,
        config: Dict[str, Any],
        device_id: str,
        type: str = "Device",
        label: str = "Device",
    ) -> None:
        
        super().__init__(config, type=type, label=label)
        self.device_id = device_id

        # base classes should implement initialization

        # register sub components

    def get_id(self):
        return self.device_id

    @abstractmethod
    def first_state(self, state: int, df: pd.DataFrame) -> None:
        """Initiualize the value for the first state.
        state = 0
        """
        ...

    @abstractmethod
    def step(self, state: int, next_state: int, df: pd.DataFrame) -> None:
        ...

    def store(self, df: pd.DataFrame, path: Path):
        path.mkdir(exist_ok=True)
        dictionary = self.config.copy()
        dictionary.pop('model', None)
        for key in ["device_class", "_path"]:
            if key in dictionary:
                del dictionary[key]
        yaml.dump(dictionary, open(path / Path("config.yaml"), "w"))
        df.to_csv(path / Path("states.csv"))
