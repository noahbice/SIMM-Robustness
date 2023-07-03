import os

import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import rcParams
from os import listdir
# plt.rc('axes', axisbelow=True)
# plt.rc('axes', facecolor='whitesmoke')
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = "serif"
rcParams['font.sans-serif'] = "Computer Modern"

colors = ['mediumorchid', 'deepskyblue', 'goldenrod', 'forestgreen',
          'violet', 'chocolate', 'salmon',  'gold', 'darkgrey', 'blue']

target_vols = [0.46, 1.02, 0.96, 0.28, 0.39, 0.33, 0.33, 0.09]

def read_csv(filename):
    on = False
    percent_vols = []
    ptvs_added = []
    percent_doses = []

    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if 'ptv' in ' '.join(row).lower():
                current_ptv = ' '.join(row).lower()
                percent_vol = []
                percent_dose = []
                on = False
            if 'relative dose [%]' in ' '.join(row).lower():
                on = True
                continue
            if on:
                try:
                    percent_dose.append(float(row[0]))
                    percent_vol_instance = float(row[-1].split(',')[-1])
                    percent_vol.append(percent_vol_instance)
                    if percent_vol_instance < 0.2:
                        on = False
                        if current_ptv not in ''.join(ptvs_added):
                            ptvs_added.append(current_ptv)
                            percent_vols.append(np.array(percent_vol))
                            percent_doses.append(np.array(percent_dose))
                except:
                    continue
    return percent_doses, percent_vols

# pd0, pv0 = read_csv('./EclipseDVHs/0.csv')
files = os.listdir('./EclipseDVHs/')


fig = plt.figure()

ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax.set_xlabel('Rel. dose [%]', fontsize=12)
ax.set_ylabel('Rel. volume [%]', fontsize=12)

gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.1)
axs = gs.subplots(sharey=True)


# fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
fig.set_size_inches(8.5, 7)
# fig.tight_layout()
fig.set_dpi(100)



for file in files:
    if '.csv' in file:
        pd, pv = read_csv('./EclipseDVHs/' + file)
        for it in range(8):
            if '0' in file:
                axs[int(it / 3), it % 3].plot(pd[it], pv[it], ls='-', linewidth=2, color=colors[it])
            else:
                axs[int(it / 3), it % 3].plot(pd[it], pv[it], ls='--', color=colors[it])

            it += 1




for it in range(8):
    axs[int(it / 3), it % 3].grid(ls='--')
    axs[int(it / 3), it % 3].set_xlim([75, 125])
    axs[int(it / 3), it % 3].set_ylim([0, 105])
    axs[int(it / 3), it % 3].set_title('PTV {}'.format(it + 1))
    axs[int(it / 3), it % 3].set_facecolor('whitesmoke')
    axs[int(it / 3), it % 3].tick_params(axis='both', which='major', labelsize=8, direction="in")
    axs[int(it / 3), it % 3].tick_params(axis='both', which='minor', labelsize=8, direction="in")
    axs[int(it / 3), it % 3].text(80, 20, '$V_{tar} = $' + str(target_vols[it]) + ' cm$^3$', color='black',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    if it not in [2, 5, 7]:
        axs[it % 3, int(it / 3)].set_xticklabels([])
    # if int(it / 3) != 0:
    #     axs[it % 3, int(it / 3)].set_xticks([])
axs[2, 2].remove()
plt.savefig('./newfigs/EclipseDVHs.png', dpi=300)
plt.show()


x = np.array([1, 2])
y = x**2
fig, ax = plt.subplots()
fig.set_size_inches(7, 7)
# fig.tight_layout()
fig.set_dpi(100)
ax.plot(x, y, color='black', ls='-', label=r'Nominal scenario $x = 0$')
ax.plot(x, y, color='black', ls='--', label=r'Translation $x, y, z = \pm 1.2$ mm')
legend = plt.legend(facecolor='white', framealpha=1, frameon=1, loc=(0.75, 0.85), fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')
plt.show()


