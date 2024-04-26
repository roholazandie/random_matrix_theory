import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from eigenfish import Eigenfish

rcParams['text.usetex'] = True
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amsfonts}"
# %config InlineBackend.figure_format = 'retina'



mdim = 5
population = np.array([0.,-1.j,1.j,1.,0.5])
r = 20
n_matrix = 100000
list_of_figure = []

fig, axs = plt.subplots(3, 3, figsize=(20,20))
fig.set_facecolor("#f4f0e8")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

for ax in axs.reshape(-1):
    for spine in ['top', 'right','left','bottom']:
        ax.spines[spine].set_visible(False)
    matrix = np.random.choice(population, (mdim,mdim))+0.j
    var_indices = np.unravel_index(np.random.choice(np.arange(0, mdim**2), 2, replace=False), (mdim,mdim))
    eigenfish = Eigenfish(matrix, var_indices)
    list_of_figure.append(eigenfish)
    eigenvalues = eigenfish.eigvals_random_ts_rect(n_matrix, r)
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), c="#383b3e", s=0.03, linewidths= 0.0001, alpha=1.)
    #ax.set_title(eigenfish.create_latex_title_rect(-r,r), fontsize=12)
    ax.set_title(eigenfish.create_simple_latex_title_rect(-r,r), fontsize=14)
    ax.set_aspect('equal', 'box')
    if np.max(np.real(eigenvalues)>10):
        ax.set_xlim([-8,8])
    #ax.set_ylim([-7,7])
    ax.set_axis_off()
plt.tight_layout()
plt.annotate("Simone Conradi, 2023", (1200.,10.), xycoords="figure points", fontsize=20)
plt.show()


