# Copyright 2026 Nardi Xhepi.
#
# This file is part of LFM3-TS, distributed under the LFM Open License v1.0: see LICENSE and NOTICE at the root
# of the repository. Unless required by applicable law or agreed to in writing, this software is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Feature extractor for LFM3-TS: normalizes each series and cuts it into patches."""

from dataclasses import dataclass

import numpy as np
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin


@dataclass
class Lfm3TsSeriesStats:
    """What normalization removed from a series; the processor writes it into the prompt so values stay recoverable."""

    offset: float
    scale: float
    length: int
    num_patches: int
    minimum: float
    maximum: float
    first: float
    last: float


class Lfm3TsFeatureExtractor(FeatureExtractionMixin):
    r"""
    Turns univariate series into the `input_values` and `input_values_mask` of an LFM3-TS model.

    Each series is standardized with its own mean and standard deviation over its observed values, then cut into
    patches of `patch_size` time steps. NaN marks a missing value: it is zeroed and masked out, as is the padding
    that fills the last patch and aligns series of different lengths.

    Args:
        patch_size (`int`, *optional*, defaults to 8):
            The number of time steps per patch. Must equal the model's `ts_config.patch_size`.
        eps (`float`, *optional*, defaults to 1e-06):
            Series whose standard deviation is below `eps` times their magnitude are treated as constant and left
            unscaled.
    """

    model_input_names = ["input_values", "input_values_mask"]

    def __init__(self, patch_size: int = 8, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.eps = eps

    def get_num_patches(self, length: int) -> int:
        """Number of patches covering `length` time steps."""
        return -(-length // self.patch_size)

    def preprocess(self, timeseries: list) -> tuple[np.ndarray, np.ndarray, list[Lfm3TsSeriesStats]]:
        """Normalizes and patches a list of 1D series.

        Returns:
            The values and the observation mask, both of shape `(num_series, max_patches, patch_size)`, and the
            statistics of each series.
        """
        normalized, stats = [], []
        for index, series in enumerate(timeseries):
            series = np.asarray(series, dtype=np.float64)
            if series.ndim != 1:
                raise ValueError(f"Series {index} must be one-dimensional, got shape {series.shape}.")
            observed = np.isfinite(series)
            if not observed.any():
                raise ValueError(f"Series {index} has no observed value.")
            # trailing missing values carry no information: the series ends at its last observed step
            length = int(np.flatnonzero(observed)[-1]) + 1
            series, observed = series[:length], observed[:length]

            offset = float(series[observed].mean())
            scale = float(series[observed].std())
            if scale <= self.eps * max(1.0, abs(offset)):
                scale = 1.0
            values = np.where(observed, (series - offset) / scale, 0.0)
            normalized.append((values, observed))
            seen = series[observed]
            stats.append(
                Lfm3TsSeriesStats(
                    offset=offset,
                    scale=scale,
                    length=length,
                    num_patches=self.get_num_patches(length),
                    minimum=float(seen.min()),
                    maximum=float(seen.max()),
                    first=float(seen[0]),
                    last=float(seen[-1]),
                )
            )

        max_patches = max(stat.num_patches for stat in stats)
        input_values = np.zeros((len(stats), max_patches * self.patch_size), dtype=np.float32)
        input_values_mask = np.zeros_like(input_values)
        for index, (values, observed) in enumerate(normalized):
            input_values[index, : len(values)] = values
            input_values_mask[index, : len(values)] = observed
        shape = (len(stats), max_patches, self.patch_size)
        return input_values.reshape(shape), input_values_mask.reshape(shape), stats

    def __call__(self, timeseries: list, return_tensors: str | None = None, **kwargs) -> BatchFeature:
        """
        Args:
            timeseries (`list`):
                One-dimensional series (arrays or lists of numbers), possibly of different lengths.
            return_tensors (`str`, *optional*):
                The tensor type to return, e.g. `"pt"`.
        """
        input_values, input_values_mask, _ = self.preprocess(timeseries)
        data = {"input_values": input_values, "input_values_mask": input_values_mask}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Lfm3TsFeatureExtractor", "Lfm3TsSeriesStats"]
