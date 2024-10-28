import collections
import random

import numpy as np
import torch

from modals.ops import cosine

PARAMETER_MAX = 10


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    x = float(level) * maxval / PARAMETER_MAX

    return torch.as_tensor(x)


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, sample, samples_pool):
        return self.f(sample, samples_pool)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def transformer(self, probability, magnitude):
        def return_function(sample, samples_pool):
            res = False
            s = []
            if random.random() < probability:
                img, s = self.xform(sample, samples_pool, magnitude)
                res = True
            return (sample, s), res

        name = self.name + '({:.1f},{})'.format(probability, magnitude)
        return TransformFunction(return_function, name)

    def do_transform(self, img, label_img_pool, magnitude):
        f = self.transformer(PARAMETER_MAX, magnitude)
        return f(img, label_img_pool)


def _interpolate(sample, class_dict, magnitude):
    """
    Interpolates between a sample and a chosen sample from the pool of images in class_dict according to magnitude.
    Args:
        sample:
        class_dict:
        magnitude:

    Returns:

    """
    magnitude = float_parameter(magnitude, 1)
    p = class_dict['weights']
    pool = class_dict['pool']

    if len(p) < 1:
        return sample, []

    # Choose how many candidates from the pool to consider, for computational efficiency
    num_candidates = max(1, int(len(pool) * 0.05))

    # Choose num_candidates candidates from the pool according to the weights
    idxs_candidates = np.random.choice(pool, num_candidates, replace=False, p=p)

    # Choose the candidate that is closest to the sample, assume that everything is on the same device
    distances = cosine(pool[idxs_candidates] - class_dict["mean"].unsqueeze(0), sample.view(-1) - class_dict["mean"])

    # Choose the candidate that is closest to the sample
    idx = idxs_candidates[torch.argmax(distances)]

    chosen = pool[idx]

    # Interpolate between the sample and the chosen candidate
    interpolated = (chosen - sample) * magnitude + sample

    return interpolated, [idx]


interpolate = TransformT('Interpolate', _interpolate)


def _extrapolate(sample, class_dict, magnitude):
    """
    Extrapolates from a sample according to magnitude.
    Args:
        sample:
        class_dict:
        magnitude:

    Returns:

    """
    magnitude = float_parameter(magnitude, 1)
    mu = class_dict['mean']
    extrapolated = (sample - mu) * magnitude + sample
    return extrapolated, []


extrapolate = TransformT('Extrapolate', _extrapolate)


def _linearpolate(sample, class_dict, magnitude):
    """
    Linearly interpolates between two samples from the pool of images in class_dict according to magnitude.
    Args:
        sample:
        class_dict:
        magnitude:

    Returns:

    """
    magnitude = float_parameter(magnitude, 1)
    pool = class_dict['pool']

    if len(pool) < 2:
        return sample, [0, 0]

    idx1, idx2 = random.sample(range(len(pool)), 2)
    y1, y2 = pool[idx1], pool[idx2]

    interpolated = (y1 - y2) * magnitude + sample

    return interpolated, [idx1, idx2]


linear_polate = TransformT('LinearPolate', _linearpolate)


def _resample(sample, class_dict, magnitude):
    """
    Resamples from a normal distribution with standard deviation from the class_dict according to magnitude.
    this is actually equivalent to choosing a random sample from the pool of samples in the class_dict and noising it.
    Maybe should be called noise instead of resample. (or resample with noise)
    Args:
        sample:
        class_dict:
        magnitude:

    Returns:

    """
    magnitude = float_parameter(magnitude, 1)
    noise = torch.randn(sample.size())
    resampled = sample + noise * class_dict['sd'] * magnitude
    return resampled, []


resample = TransformT('Resample', _resample)


HP_TRANSFORMS = [
    interpolate,
    extrapolate,
    linear_polate,
    resample
]

NAME_TO_TRANSFORM = collections.OrderedDict((t.name, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)
