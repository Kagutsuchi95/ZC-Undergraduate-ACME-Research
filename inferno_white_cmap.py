"""
inferno_white: a variant of matplotlib's 'inferno' colormap where the
near-black floor is replaced with white, so low-intensity noise near zero
stays visible instead of disappearing into the background.

Usage:
    from inferno_white_cmap import inferno_white
    plt.imshow(data, cmap=inferno_white)
"""

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def make_inferno_white(skip_frac=0.12, n_blend=20):
    """
    Build the inferno_white colormap.

    skip_frac : float
        Fraction of inferno's darkest samples to discard (default 0.12).
        Increase this if you still see too much darkness near zero;
        decrease it if the transition to white feels too abrupt.
    n_blend : int
        Number of samples used to blend white into the first retained
        inferno color. Higher = smoother, more gradual transition.
    """
    inferno = mpl.colormaps['inferno'].resampled(256)
    colors = inferno(np.linspace(0, 1, 256))[:, :3]

    skip = int(skip_frac * 256)
    body = colors[skip:]

    white = np.array([[1.0, 1.0, 1.0]])
    blend = np.linspace(white, body[0:1], n_blend).reshape(n_blend, 3)

    new_colors = np.vstack([blend, body])
    return LinearSegmentedColormap.from_list('inferno_white', new_colors, N=256)

def make_inverse_inferno_white(skip_frac=0.12, n_blend=20):
    """
    Build the inverse_inferno_white colormap.

    skip_frac : float
        Fraction of inferno's darkest samples to discard (default 0.12).
        Increase this if you still see too much darkness near zero;
        decrease it if the transition to white feels too abrupt.
    n_blend : int
        Number of samples used to blend white into the first retained
        inferno color. Higher = smoother, more gradual transition.
    """
    inferno = mpl.colormaps['inferno'].resampled(256)
    colors = inferno(np.linspace(0, 1, 256))[:, :3]

    skip = int(skip_frac * 256)
    body = colors[skip:]

    white = np.array([[1.0, 1.0, 1.0]])
    blend = np.linspace(white, body[0:1], n_blend).reshape(n_blend, 3)

    new_colors = np.vstack([blend, body])
    
    # Inverse cmap
    new_colors = new_colors[::-1]
        
    return LinearSegmentedColormap.from_list('inferno_white', new_colors, N=256)


inferno_white = make_inferno_white()
inverse_inferno_white = make_inverse_inferno_white()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x, y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    z = np.exp(-(x**2 + y**2)) + 0.3 * np.exp(-((x - 1.5)**2 + (y - 1.5)**2) / 0.3)

    plt.imshow(z, cmap=inferno_white)
    plt.colorbar()
    plt.title('inferno_white demo')
    plt.show()