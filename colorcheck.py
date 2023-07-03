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

ss = np.linspace(0, 3, 100)
y1 = ss**2
y2 = ss ** 2.5 + 0.1
y1e = 2* np.sqrt(y1)
y2e = 0.8 * np.sqrt(y2)


bars = '#567dbd'
line = '#5bedf5'

plt.plot(ss, y2, color=line, ls='-', label=r'Analytical')
plt.plot(ss, y1, color='firebrick', label=r'Numerical ')
plt.plot(ss, y2 - y2e, color=bars, ls='--', label=r'Analytical')
plt.plot(ss, y2 + y2e, color=bars, ls='--')
plt.fill_between(ss, y1-y1e, y1+y1e, color='lightcoral', label=r'Numerical')

# legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.06, 0.5), fontsize=12)
# legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.06, 0.06), fontsize=12)
legend = plt.legend(facecolor='white', framealpha=1, frameon=1, fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')
plt.grid(ls='--')

plt.show()