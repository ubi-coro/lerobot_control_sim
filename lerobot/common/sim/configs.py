import abc
from dataclasses import dataclass, field
from typing import List
import draccus

@dataclass
class SimConfig(draccus.ChoiceRegistry, abc.ABC):
    env: str
    simulated_arms: List[str]
    calibration_dir: str

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@SimConfig.register_subclass("aloha")
@dataclass
class AlohaSimConfig(SimConfig):
    env: str = "aloha"
    simulated_arms: List[str] = field(default_factory=lambda: [
        "left_follower",
        "right_follower"])
    calibration_dir: str = ".cache/calibration/aloha_sim"
