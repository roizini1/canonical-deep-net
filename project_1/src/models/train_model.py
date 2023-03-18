# loggers imports:
import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
# db imports:
from torchvision.datasets import MNIST

from conf.config import Config_class

# pytorch related imports:
from lightning_net import Net

# config store for hydra:
cs = ConfigStore.instance()
cs.store(name="Model_config", node=Config_class)


@hydra.main(version_base=None, config_path="./conf", config_name="model_config")
def my_app(cfg: Config_class) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.trained_model.final_models_dir,
        filename='checkpoint_{epoch:02d}-{loss:.2f}',
        monitor='loss',
        save_last=True,
        save_top_k=1,
        mode='min',
    )

    stop_callback = EarlyStopping(
        monitor='loss',
        patience=5,
        mode='min',  # for this loss mode is min
    )

    callback_list = [checkpoint_callback, RichProgressBar(), stop_callback, ]
    tb_logger = TensorBoardLogger(save_dir=cfg.trained_model.tb_save_dir)

    trainer = Trainer(
        accelerator="cpu",
        devices="auto",
        max_epochs=cfg.training_process.max_epochs,
        callbacks=callback_list,
        logger=tb_logger,
    )
    transform_X = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_Y = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # model definition
    model = Net(cfg.training_process)

    # datasets definition
    trainX_ds = MNIST(cfg.db.PATH_DATASETS, train=True, download=True, transform=transform_X)
    trainY_ds = MNIST(cfg.db.PATH_DATASETS, train=True, download=True, transform=transform_Y)

    # dataloaders
    trainX_loader = DataLoader(trainX_ds, batch_size=cfg.training_process.batch_size, num_workers=6)  # , shuffle=True
    trainY_loader = DataLoader(trainY_ds, batch_size=cfg.training_process.batch_size, num_workers=6)  # , shuffle=True

    trainer.fit(model, [trainX_loader, trainY_loader])


if __name__ == "__main__":
    my_app()
