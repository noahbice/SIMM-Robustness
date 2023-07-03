import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = "serif"
rcParams['font.sans-serif'] = "Computer Modern"

colors = ['mediumorchid', 'deepskyblue', 'goldenrod', 'forestgreen',
          'violet', 'chocolate', 'salmon',  'gold', 'darkgrey', 'blue']
plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')

def intersection(r, m, d):
    R = r + m
    term1 = (np.pi / (12*d))
    term2 = (R + r - d)**2
    term3 = (d**2 + 2*d*r - 3*r**2 + 2*d*R + 6*r*R - 3*R**2)
    return term1 * term2 * term3

def coverage(r, m, d):
    v_int = intersection(r, m, d)
    v_tar = (4 / 3) * np.pi * r**3
    return v_int / v_tar


def plot_coverage(r, m, color, ls='-'):
    D_1 = np.linspace(0, m, 2)
    C_1 = np.array([1, 1])
    D_0 = np.linspace(2*r + m, 5*r + m, 2)
    C_0 = np.array([0, 0])
    D = np.linspace(m, 2*r + m, 100)

    plt.plot(D_1, C_1, color)
    plt.plot(D_0, C_0, color)
    plt.plot(D, coverage(r, m, D), color, ls=ls, label="$r={}, m={}$".format(r,m))


if True:
    # plot_coverage(20, 1, 'blue')
    # plot_coverage(20, 3, 'black')
    plot_coverage(10, 1, 'red', '--')
    plot_coverage(10, 3, 'black', '--')
    plot_coverage(5, 1, 'red')
    plot_coverage(5, 3, 'black')
    plt.xlabel("$x$ [arbitrary units]", fontsize=15)
    plt.ylabel(r"$V_{100\%}$", fontsize=15)
    legend = plt.legend(facecolor='white', framealpha=1, frameon=1, fontsize=12)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.grid(ls="--")
    plt.xlim([0, 25])
    plt.ylim([-0.05, 1.05])
    plt.savefig('./newfigs/v100analytical.png', dpi=300)
    plt.show()



p = np.array([])

