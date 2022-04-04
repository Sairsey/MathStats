import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as stats

import os

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor,
                      edgecolor='midnightblue', **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def task5():
  sizes = [20, 60, 100]
  rhos = [0, 0.5, 0.9]
  
  if not os.path.exists("task5_data"):
    os.mkdir("task5_data")
  
  # Count params for table
  # count for 2d normal
  for size in sizes:
    rows = []  
    for rho in rhos:
      rows.append(['$\\rho = ' + str(rho) + '~\eqref{eq:correlation}$', '$r ~\eqref{eq:pearson}$',
             '$r_Q ~\eqref{eq:rq}$', '$r_S ~\eqref{eq:spearman}$'])
      Pirson = []
      Spirman = []
      SquaredCorrelation = []

      for repeat in range(0, 1000):
        sample_d = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)

        cov = np.cov(sample_d[:, 0], sample_d[:, 1])
        Pirson.append(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

        n1, n2, n3, n4 = 0, 0, 0, 0
        for el in sample_d:
          if el[0] >= 0 and el[1] >= 0:
            n1 += 1
          elif el[0] < 0 and el[1] >= 0:
            n2 += 1
          elif el[0] < 0 and el[1] < 0:
            n3 += 1
          else:
            n4 += 1

        Spirman.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1])[0])
        SquaredCorrelation.append((n1 - n2 + n3 - n4) / len(sample_d))
      rows.append(['$E(z)$', np.mean(Pirson), np.mean(Spirman), np.mean(SquaredCorrelation)])
      rows.append(['$E(z^2)$', np.mean(np.asarray([el * el for el in Pirson])),
          np.mean(np.asarray([el * el for el in Spirman])),
          np.mean(np.asarray([el * el for el in SquaredCorrelation]))])
      rows.append(['$D(z)$', np.std(Pirson), np.std(Spirman), np.std(SquaredCorrelation)])
      if rho != rhos[-1]:
        rows.append(['', '', '', ''])
    with open("task5_data/task5_" + str(size) + ".txt", "w") as f:
      f.write("\\begin{tabular}{|c|c|c|c|}\n")
      f.write("\\hline\n")
      for row in rows:
        f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
        f.write("\\hline\n")
      f.write("\\end{tabular}")

  # count for mix normal
  for size in sizes:
    rows = []
    rows.append(['$\\n = ' + str(size) + '~\eqref{eq:correlation}$', '$r ~\eqref{eq:pearson}$',
           '$r_Q ~\eqref{eq:rq}$', '$r_S ~\eqref{eq:spearman}$'])
    Pirson = []
    Spirman = []
    SquaredCorrelation = []

    for repeat in range(0, 1000):
      sample_d = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) +\
                     0.1 * np.random.multivariate_normal([0, 0], [[10, 0.9], [0.9, 10]], size=size)

      cov = np.cov(sample_d[:, 0], sample_d[:, 1])
      Pirson.append(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

      n1, n2, n3, n4 = 0, 0, 0, 0
      for el in sample_d:
        if el[0] >= 0 and el[1] >= 0:
          n1 += 1
        elif el[0] < 0 and el[1] >= 0:
          n2 += 1
        elif el[0] < 0 and el[1] < 0:
          n3 += 1
        else:
          n4 += 1

      Spirman.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1])[0])
      SquaredCorrelation.append((n1 - n2 + n3 - n4) / len(sample_d))
    rows.append(['$E(z)$', np.mean(Pirson), np.mean(Spirman), np.mean(SquaredCorrelation)])
    rows.append(['$E(z^2)$', np.mean(np.asarray([el * el for el in Pirson])),
        np.mean(np.asarray([el * el for el in Spirman])),
        np.mean(np.asarray([el * el for el in SquaredCorrelation]))])
    rows.append(['$D(z)$', np.std(Pirson), np.std(Spirman), np.std(SquaredCorrelation)])
    if rho != rhos[-1]:
      rows.append(['', '', '', ''])
  with open("task5_data/task5_mix.txt", "w") as f:
    f.write("\\begin{tabular}{|c|c|c|c|}\n")
    f.write("\\hline\n")
    for row in rows:
      f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
      f.write("\\hline\n")
    f.write("\\end{tabular}")

  # Draw ellipses
  for size in sizes:
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, rho in zip(axs, rhos):
      sample_d = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)
      ax.scatter(sample_d[:, 0], sample_d[:, 1], color='red', s=3)
      ax.set_xlim(min(sample_d[:, 0]) - 2, max(sample_d[:, 0]) + 2)
      ax.set_ylim(min(sample_d[:, 1]) - 2, max(sample_d[:, 1]) + 2)
      print(min(sample_d[:, 0]), max(sample_d[:, 0]))
      print(min(sample_d[:, 1]), max(sample_d[:, 1]))
      ax.axvline(c='grey', lw=1)
      ax.axhline(c='grey', lw=1)
      title = r'n = ' + str(size) + r', $\rho$  = ' + str(rho)
      ax.set_title(title)
      confidence_ellipse(sample_d[:, 0], sample_d[:, 1], ax)
    fig.savefig('task5_data/' + str(size) + '.png', dpi=200)
    
  # Draw mixed ellipse
  fig, axs = plt.subplots(1, 3, figsize=(9, 3))
  for ax, size in zip(axs, sizes):
      sample_d = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) +\
                 0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], size=size)
      ax.scatter(sample_d[:, 0], sample_d[:, 1], color='red', s=3)
      ax.set_xlim(min(sample_d[:, 0]) - 2, max(sample_d[:, 0]) + 2)
      ax.set_ylim(min(sample_d[:, 1]) - 2, max(sample_d[:, 1]) + 2)
      print(min(sample_d[:, 0]), max(sample_d[:, 0]))
      print(min(sample_d[:, 1]), max(sample_d[:, 1]))
      ax.axvline(c='grey', lw=1)
      ax.axhline(c='grey', lw=1)
      title = r'mixed: n = ' + str(size)
      ax.set_title(title)
      confidence_ellipse(sample_d[:, 0], sample_d[:, 1], ax)
  fig.savefig('task5_data/mix.png', dpi=200)
  plt.show()

if __name__ == "__main__":
  task5()