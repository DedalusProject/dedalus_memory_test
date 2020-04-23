"""
Plot planes from joint analysis files.

Usage:
    plot_memory.py <filename> [--output=<image>] [--total]

Options:
    --output=<image>  Output filename [default: ./memory.pdf]
    --total           Plot total

"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.getcwd())
from test_params import *


def main(filename, output, total=False):
    data = np.loadtxt(filename) / 1e9
    plt.figure()
    if total:
        plt.plot(np.sum(data, axis=1), '--k')
    plt.plot(data)
    plt.ylim(0, None)
    plt.xlabel("Measurement")
    plt.ylabel("Memory (GB)")
    plt.title(test_title)
    plt.savefig(output)


if __name__ == "__main__":
    import pathlib
    from docopt import docopt

    args = docopt(__doc__)
    filename = args['<filename>']
    output = pathlib.Path(args['--output']).absolute()
    total = args['--total']
    main(filename, output=output, total=total)

