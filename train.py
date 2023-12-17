"""
This file is expecting the 'pipeline' package name to be defined
This file completely defines the experiment to run
"""
import argparse
import contextlib
from typing import Tuple, Generator, Optional, Dict, List

import torch
import torch.utils.data

from lib import data
from lib import logger
from lib.discriminators import DCDiscriminator
from lib.gan import GAN
from lib.generators import DCGenerator
from lib.metrics import *
from lib.normalization import apply_normalization, SpectralNormalizer
from lib.predicates import TrainPredicate, IgnoreFirstNEpochsPredicate, EachNthEpochPredicate
from lib.storage import ExperimentsStorage
from lib.train import Stepper, WganEpochTrainer, GanTrainer
from lib.wandb_logger import WandbCM


def init_storage() -> ExperimentsStorage:
    # === config variables ===
    experiments_dir = 'experiments'
    checkpoint_filename = './training_checkpoint'
    model_state_filename = './model_state'
    # ========================
    return ExperimentsStorage(experiments_dir=experiments_dir, checkpoint_filename=checkpoint_filename,
                              model_state_filename=model_state_filename)


experiments_storage = init_storage()


def form_metric() -> Metric:
    return MetricsSequence(
        CriticValuesDistributionMetric(values_cnt=1000),
        GeneratedImagesMetric(5, 5),
    )


def form_metric_predicate() -> Optional[TrainPredicate]:
    return IgnoreFirstNEpochsPredicate(20) & EachNthEpochPredicate(10)


def form_dataset(data_dir: str, train: bool = False) -> torch.utils.data.Dataset:
    return data.UnifiedDatasetWrapper(data.get_simple_images_dataset(data_dir, train, val_ratio=0.05))


def init_logger(model_name: str = ''):
    project_name = 'GAN-pokemons'
    config = logger.get_default_config()
    @contextlib.contextmanager
    def logger_cm():
        try:
            with WandbCM(project_name=project_name, experiment_id=model_name, config=config) as wandb_logger:
                yield wandb_logger
        finally:
            pass
    return logger_cm


def form_gan_trainer(data_dir: str, model_name: str,
                     gan_model: Optional[GAN] = None, n_epochs: int = 100,
                     enable_logging: bool = True) -> Generator[Tuple[int, GAN], None, GAN]:
    """
    :return: a generator that yields (epoch number, gan_model after this epoch)
    """
    logger_cm_fn = init_logger(model_name) if enable_logging else None
    metric = form_metric()
    metric_predicate = form_metric_predicate()

    train_dataset = form_dataset(data_dir, train=True)
    val_dataset = form_dataset(data_dir, train=False)

    # for local testing
    # val_size = int(0.1 * len(val_dataset))
    # val_dataset = torch.utils.data.Subset(val_dataset, np.arange(val_size))
    # -------
    noise_dimension = 100

    def uniform_noise_generator(n: int) -> torch.Tensor:
        return 2*torch.rand(size=(n, noise_dimension)) - 1  # [-1, 1]

    latent_channels = 1024
    image_channels = 3

    generator = DCGenerator(noise_dim=noise_dimension, latent_channels=latent_channels, image_channels=image_channels)
    discriminator = DCDiscriminator(latent_channels=latent_channels, image_channels=image_channels)
    discriminator = apply_normalization(discriminator, SpectralNormalizer)

    regularizer = None

    if gan_model is None:
        gan_model = GAN(generator, discriminator, uniform_noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-3)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    )

    epoch_trainer = WganEpochTrainer(n_critic=1, batch_size=32)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=False, save_checkpoint_once_in_epoch=0)
    train_gan_generator = trainer.train(gan_model=gan_model,
                                        train_dataset=train_dataset, val_dataset=val_dataset,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        metric=metric, metric_predicate=metric_predicate,
                                        logger_cm_fn=logger_cm_fn,
                                        regularizer=regularizer)
    return train_gan_generator


def run() -> GAN:
    args = argparse.ArgumentParser(description="GAN training script")
    args.add_argument(
        "-i",
        "--input",
        type=str,
        help="a directory with images",
    )
    args.add_argument(
        "-l",
        action="store_true",
        help="enable logging",
    )
    args = args.parse_args()

    model_name = 'test'
    gan_trainer = form_gan_trainer(data_dir=args.input, model_name=model_name, n_epochs=150,
                                   enable_logging=args.l)
    gan = None
    for epoch, gan in gan_trainer:
        pass

    # evaluate model somehow ...
    return gan


if __name__ == '__main__':
    run()
