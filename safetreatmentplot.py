import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)
plt.rc('axes', facecolor='whitesmoke')

plt.title('Total error $\delta r$', fontsize=14)
plt.xlabel(r'$\delta x$ [mm]', fontsize=14)
plt.ylabel(r'$R \delta \theta$ [mm]', fontsize=14)

tolerances = [0.5, 1.0, 1.5, 2.0]
xs = np.linspace(0, 2, 500)
ys = np.linspace(0, 2, 500)

xx, yy = np.meshgrid(xs, ys)
total_error = np.sqrt(xx**2 + yy**2)
plt.imshow(total_error[::-1], extent=[0, 2, 0, 2])
plt.colorbar(label='Shift magnitude [mm]')
for tol in tolerances:
    plt.plot(tol*np.cos(xs), tol*np.sin(xs), color='black')
plt.xticks(np.linspace(0, 2, 5))
plt.yticks(np.linspace(0, 2, 5))
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.show()

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

Rcontours = [30, 70, 100]
for tol in tolerances:
    fig, axs = plt.subplots()
    dxs = np.linspace(0.01, tol - 0.01, 500)
    dthetas = np.linspace(0.01, 2, 500)
    xx, yy = np.meshgrid(dxs, dthetas)
    axs.set_title(r'Maximum acceptable target-to-isocenter distance for $\delta r < {}$ mm'.format(tol), fontsize=14)
    axs.set_xlabel(r'Translation error $\delta x$ [mm]', fontsize=14)
    axs.set_ylabel(r'Rotation error $\delta \theta$ [degrees]', fontsize=14)
    R = np.sqrt((tol**2 - xx**2) / (yy * (np.pi / 180))**2)
    im = axs.imshow(R[::-1], extent=[0.01, tol - 0.01, 0.01, 2], vmin=0, vmax=110)
    for r in Rcontours:
        axs.plot(dxs, (180 / np.pi) * np.sqrt((tol**2 - dxs**2) / r ** 2), color='black')
    fig.colorbar(im, label='Maximum target-to-iso distance R [mm]')
    axs.set_xlim([0.01, tol - 0.01])
    axs.set_yticks(np.linspace(0, 2, 5))
    axs.set_ylim([0.01, 2])
    forceAspect(axs, aspect=1)
    plt.show()

