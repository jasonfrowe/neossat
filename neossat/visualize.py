import numpy as np

from photutils import CircularAperture

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_histogram(scidata, imstat, sigscalel, sigscaleh):
    """"""

    matplotlib.rcParams.update({'font.size': 24})  # Adjust font.

    flat = scidata.flatten()
    vmin = np.min(flat[flat > imstat[2] - imstat[3]*sigscalel])
    vmax = np.max(flat[flat < imstat[2] + imstat[3]*sigscaleh])

    plt.figure(figsize=(12, 6))  # Adjust size of figure.
    image_hist = plt.hist(scidata.flatten(), 100, range=(vmin, vmax))
    plt.xlabel('Image Counts (ADU)')
    plt.ylabel('Number Count')
    plt.show()

    return


def plot_image_wsource(scidata, imstat, sigscalel, sigscaleh, sources=None, xy=None, figname=None, display=True):
    """"""

    eps = 1.0e-9
    sigscalel = -np.abs(sigscalel)  # Expected to be negative.
    sigscaleh = np.abs(sigscaleh)  # Expected to be positive.

    matplotlib.rcParams.update({'font.size': 24})  # Adjust font.

    flat = scidata.flatten()
    vmin = np.min(flat[flat > imstat[2] + imstat[3]*sigscalel]) - imstat[0] + eps
    vmax = np.max(flat[flat < imstat[2] + imstat[3]*sigscaleh]) - imstat[0] + eps

    if sources is not None:
        positions = np.column_stack([sources['xcentroid'], sources['ycentroid']])
    elif xy is not None:
        positions = np.column_stack(xy)
    else:
        raise ValueError('Either sources or xy must be give.')

    apertures = CircularAperture(positions, r=4.)

    plt.figure(figsize=(20, 20))  # Adjust size of figure.
    imgplot = plt.imshow(scidata - imstat[0], norm=LogNorm(), vmin=vmin, vmax=vmax)  # TODO scidata index?
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    for i in range(len(positions)):
        plt.annotate('{}'.format(i), positions[i])
    plt.axis((-0.5, scidata.shape[1]-0.5, -0.5, scidata.shape[0]-0.5))
    plt.xlabel("Column (Pixels)")
    plt.ylabel("Row (Pixels)")

    if figname is not None:
        plt.savefig(figname)

    if display:
        plt.show()

    plt.close()

    return


def plot_image(scidata, imstat, sigscalel, sigscaleh):
    """"""

    eps = 1.0e-9
    sigscalel = -np.abs(sigscalel)  # Expected to be negative.
    sigscaleh = np.abs(sigscaleh)  # Expected to be positive.

    matplotlib.rcParams.update({'font.size': 24})  # Adjust font.

    flat = scidata.flatten()
    vmin = np.min(flat[flat > imstat[2] + imstat[3] * sigscalel]) - imstat[0] + eps
    vmax = np.max(flat[flat < imstat[2] + imstat[3] * sigscaleh]) - imstat[0] + eps

    plt.figure(figsize=(20, 20))  # Adjust size of figure.
    imgplot = plt.imshow(scidata[:, :] - imstat[0], norm=LogNorm(), vmin=vmin, vmax=vmax)
    plt.axis((0, scidata.shape[1], 0, scidata.shape[0]))
    plt.xlabel("Column (Pixels)")
    plt.ylabel("Row (Pixels)")
    plt.show()

    return
