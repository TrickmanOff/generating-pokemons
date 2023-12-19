"""
This file is expecting the 'pipeline' package name to be defined
This file completely defines the experiment to run
"""
import argparse
import contextlib
from typing import Tuple, Generator, Optional, Dict, List, Callable

import torch
import torch.utils.data
from torchvision import transforms

from lib import data
from lib import logger
from lib.discriminators import FixedDCDiscriminator, Discriminator as DiscriminatorModel
from lib.gan import GAN
from lib.generators import FixedDCGenerator, Generator as GeneratorModel
from lib.metrics import *
from lib.normalization import apply_normalization, SpectralNormalizer, ABCASNormalizer
from lib.predicates import TrainPredicate, IgnoreFirstNEpochsPredicate, EachNthEpochPredicate
from lib.storage import ExperimentsStorage
from lib.train import Stepper, GANEpochTrainer, GanTrainer
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
        SSIMGenSimilarity(values_cnt=25),
        FIDMetric(values_cnt=2000),
        SSIMEvalMetric(values_cnt=2000),
    )


def form_per_batch_metrics():
    return [
        SSIMMetric(),
    ]


def form_metric_predicate() -> Optional[TrainPredicate]:
    return EachNthEpochPredicate(5)


def form_dataset(data_dir: str, train: bool = False) -> torch.utils.data.Dataset:
    if data_dir is None:
        dataset = data.get_cats_faces_dataset(train=train, val_ratio=0.13, load_all_in_memory=False)
    else:
        dataset = data.get_simple_images_dataset(data_dir, train=train, val_ratio=0.13, load_all_in_memory=False)
    return data.UnifiedDatasetWrapper(dataset)


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


def get_generator(noise_dim: int, arch: str = 'gan_small') -> GeneratorModel:
    latent_channels = {
        'big': 1024,
        'small': 512,
    }
    gan_training_type, model_size = arch.split('_')
    return FixedDCGenerator(noise_dim=noise_dim, latent_channels=latent_channels[model_size],
                            image_channels=3)


def get_discriminator(arch: str = 'gan_small') -> DiscriminatorModel:
    latent_channels = {
        'big': 1024,
        'small': 512,
    }
    gan_training_type, model_size = arch.split('_')
    return FixedDCDiscriminator(latent_channels=latent_channels[model_size], image_channels=3)


def get_noise_generator(noise_dim: int) -> Callable[[int, Optional[int]], torch.Tensor]:
    def uniform_noise_generator(n: int, seed=None) -> torch.Tensor:
        gen = None if seed is None else torch.manual_seed(seed)
        return 2*torch.rand(size=(n, noise_dim), generator=gen) - 1  # [-1, 1]
    return uniform_noise_generator


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

    noise_generator = get_noise_generator(noise_dimension)

    arch = 'gan_small'
    gan_training_type, model_size = arch.split('_')
    generator = get_generator(noise_dimension, arch)
    discriminator = get_discriminator(arch)
#     discriminator = apply_normalization(discriminator, SpectralNormalizer)
#     discriminator = apply_normalization(discriminator, ABCASNormalizer)

    regularizer = None

    if gan_model is None:
        gan_model = GAN(generator, discriminator, noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    )

    per_batch_metrics = form_per_batch_metrics()
    epoch_trainer = GANEpochTrainer(n_critic=1, batch_size=128, use_wgan_loss=(gan_training_type == 'wgan'), per_batch_metrics=per_batch_metrics)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=False, save_checkpoint_once_in_epoch=10)
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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="GAN training script")
    args.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="a directory with images",
    )
    args.add_argument(
        "-q",
        action="store_true",
        help="disable logging",
    )
    args = args.parse_args()
    model_name = 'test'
    gan_trainer = form_gan_trainer(data_dir=args.input, model_name=model_name, n_epochs=150,
                                   enable_logging=not args.q)
    gan = None
    for epoch, gan in gan_trainer:
        pass

    # evaluate model somehow ...
