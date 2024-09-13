# ------------------------------------------------------------------------ #
# Colored noise generator for iCEM
# ------------------------------------------------------------------------ #
# Generate colored noise (Gaussian distributed noise with a power law spectrum)
# Adapted from colorednoise package, credit: https://github.com/felixpatzelt/colorednoise
from typing import Iterable, Union

import torch
from packaging import version

def rfftfreq(samples: int, device: torch.device) -> torch.Tensor:
    if version.parse(torch.__version__) >= version.parse("1.8.0"):
        return torch.fft.rfftfreq(samples, device=device)
    freqs = np.fft.rfftfreq(samples)  # type: ignore
    return torch.from_numpy(freqs).to(device)

def powerlaw_psd_gaussian(
    exponent: float,
    size: Union[int, Iterable[int]],
    device: torch.device,
    fmin: float = 0,
):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in: Timmer, J. and Koenig, M.:On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Args:
        exponent (float): the power-spectrum of the generated noise is proportional to
            S(f) = (1 / f)**exponent.
        size (int or iterable): the output shape and the desired power spectrum is in the last
            coordinate.
        device (torch.device): device where computations will be performed.
        fmin (float): low-frequency cutoff. Default: 0 corresponds to original paper.

    Returns
        (torch.Tensor): The samples.
    """

    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, int):
        size = [size]
    else:
        size = list(size)

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we assume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples, device=device)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    ix = torch.sum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].detach().clone()
    w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * torch.sqrt(torch.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    m = torch.distributions.Normal(loc=0.0, scale=s_scale.flatten())
    sr = m.sample(tuple(size[:-1]))
    si = m.sample(tuple(size[:-1]))

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = torch.fft.irfft(s, n=samples, axis=-1) / sigma

    return y