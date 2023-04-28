from dataclasses import dataclass


@dataclass
class db_config:
    PATH_DATASETS: str
    url_1: str
    url_2: str


@dataclass
class layers:
    in_layer: int
    out_layer: int


@dataclass
class training_config:
    max_epochs: int
    batch_size: int
    lr: float
    lr_dim_reduction: float
    optimizer: str
    gates_learning: bool
    alpha: float
    sigma_x: float
    sigma_y: float
    lambda_x: float
    lambda_y: float
    layers: layers


@dataclass
class trained_model_config:
    final_models_dir: str
    tb_save_dir: str


@dataclass
class Config_class:
    db: db_config
    training_process: training_config
    trained_model: trained_model_config
