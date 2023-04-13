"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np
from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    # Create input data, run through ReLU and plot.
    relu = ReLU()
    x = np.arange(-4, 4, .1)
    plt.plot(x, relu(x))
    # END TODO###################
