from typing import Literal
from omni.isaac.lab.utils import configclass
from dataclasses import MISSING

@configclass
class CleanRLCfg:
    seed: int = MISSING
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = MISSING
    capture_video: bool = False
    env_id: str = MISSING
    total_timesteps: int = 30000000
    learning_rate: float = 0.0026
    num_envs: int = 4096
    num_steps: int = 16
    