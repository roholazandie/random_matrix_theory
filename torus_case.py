import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}"
from eigenfish import Eigenfish

mdim = 6
population = np.array([0.,0.,-1.j,1.j,0.2])
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
    eigenvalues = eigenfish.eigvals_random_ts_torus(n_matrix)
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), c="#383b3e", s=0.02, linewidths= 0.0001, alpha=1.)
    ax.set_title(eigenfish.create_latex_title_torus(), fontsize=15)
    ax.set_aspect('equal', 'box')
    #ax.set_xlim([-8,8])
    #ax.set_ylim([-8,8])
    ax.set_axis_off()
plt.tight_layout()
plt.annotate("Simone Conradi, 2023", (1100.,20.), xycoords="figure points", fontsize=12)
plt.show()