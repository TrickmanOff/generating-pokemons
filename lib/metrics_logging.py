"""A bridge between metrics and logger"""
from dataclasses import dataclass
from typing import Any, Tuple, Optional, Dict, Type, List

import numpy as np
import torch
from matplotlib import pyplot as plt

from lib.logger import GANLogger
from lib.metrics import *

"""
Agreements:
- All functions that use matplotlib.pyplot to plot a chart should not call `plt.show()`.
They are expected to create new figure and use it. When logging two options are possible:
a) plt.show()
b) plt.savefig() + plt.close()
"""


def _reduce_range(x: np.ndarray, quantile_removed: float, values_range: Optional[Tuple[float, float]] = None):
    """
    Removes at least (quantile_removed/2)*100% minimal and maximal values.
    Then only values from the range `values_range` (exclusively) are left, if this range is not None.
    """
    lower_bound = np.quantile(x, quantile_removed / 2)
    upper_bound = np.quantile(x, 1 - quantile_removed / 2)
    if values_range is not None:
        lower_bound = max(lower_bound, values_range[0])
        upper_bound = min(upper_bound, values_range[1])
    return x[(x > lower_bound) & (x < upper_bound)]


# for metrics that return samples
@dataclass
class DistributionLogInfo:
    name: str
    values_range: Optional[Tuple[float, float]] = None


def log_metric(metric: Metric, results: Any, logger: GANLogger, period: str, period_index: int) -> None:
    """
    :param metric:
    :param logger:
    """
    if isinstance(metric, TransformData):
        log_metric(metric.metric, results, logger, period=period, period_index=period_index)
    elif isinstance(metric, MetricsSequence):
        for metric, result in zip(metric.metrics, results):
            log_metric(metric, result, logger, period=period, period_index=period_index)
    elif isinstance(metric, DataStatistics):
        for statistic, result in zip(metric.statistics, results):
            log_metric(statistic, result, logger, period=period, period_index=period_index)
    elif isinstance(metric, CriticValuesDistributionMetric):
        critic_vals_true: np.ndarray
        critic_vals_gen: np.ndarray
        critic_vals_true, critic_vals_gen = results
        logger.log_critic_values_distribution(critic_vals_true, critic_vals_gen, period=period, period_index=period_index)
    elif isinstance(metric, DataStatistics):
        for statistic, res in zip(metric.statistics, results):
            log_metric(statistic, res, logger, period=period, period_index=period_index)
    elif isinstance(metric, ConditionBinsMetric):
        metric_name = metric.metric.NAME
        for bin_i, value in enumerate(results):
            logger.log_metrics(data={f'bin #{bin_i}: {metric_name}': value}, period=period,
                               period_index=period_index, commit=False)
        logger.log_metrics(data={f'bins avg: {metric_name}': np.mean(results)}, period=period,
                           period_index=period_index, commit=False)
    elif isinstance(metric, DataStatistic):
        raise NotImplementedError
    elif isinstance(metric, DiscriminatorParameterMetric) or isinstance(metric, GeneratorParameterMetric) or \
         isinstance(metric, DiscriminatorAttributeMetric) or isinstance(metric, GeneratorAttributeMetric):

        if isinstance(results, np.ndarray) or isinstance(results, torch.Tensor):
            results = results.tolist()
            if isinstance(results, list) and len(results) == 1:
                results = results[0]

        try:  # if iterable
            data = {}
            for i, val in enumerate(results):
                data[f'{metric.NAME}[{i}]'] = val
        except TypeError:
            data = {metric.NAME: results}

        if metric.NAME.endswith('coefs'):
            data[metric.NAME + ' prod'] = np.prod(results)

        logger.log_metrics(data=data, period=period, period_index=period_index, commit=False)
    elif isinstance(metric, CriticValuesStats) or isinstance(metric, GeneratorValuesStats):
        if isinstance(metric, CriticValuesStats):
            part = 'discriminator'
            prefixes = ['val', 'gen']
        else:
            part = 'generator'
            prefixes = ['gen']
            results = (results,)
        data = {}
        for stats, prefix in zip(results, prefixes):
            for stat_name, stat_val in stats.items():
                data[f'train/{part}/{prefix}_{stat_name}'] = stat_val
        logger.log_metrics(data=data, period=period, period_index=period_index, commit=False)
    elif isinstance(results, plt.Figure):
        logger.log_pyplot(metric.NAME, period, period_index, fig=results)
    else:
        raise NotImplementedError(f'Metric "{type(metric)}" is not supported for logging')
