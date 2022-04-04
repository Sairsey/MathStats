import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import chi2
import scipy.stats as stats

def get_cdf(x, mu, sigma):
    return stats.norm.cdf(x, mu, sigma)

def normal_exp():
  sz = 100
  samples = np.random.normal(0, 1, size=sz)
  mu_c = np.mean(samples)
  sigma_c = np.std(samples)
  alpha = 0.05
  p = 1 - alpha
  k = 7
  left = mu_c - 3
  right = mu_c + 3
  borders = np.linspace(left, right, num=(k - 1))
  value = chi2.ppf(p, k - 1)
  p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
  for i in range(len(borders) - 1):
    val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
    p_arr = np.append(p_arr, val)
  p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))
  print(f"mu: " + str(mu_c) + ", sigma: " + str(sigma_c))
  print(f"Промежутки: " + str(borders) + "\n"
        f"p_i: \n " + str(p_arr) + " \n"
        f"n * p_i: \n " + str(p_arr * sz) + "\n"
        f"Сумма: " + str(np.sum(p_arr)) + "\n")
  n_arr = np.array([len(samples[samples <= borders[0]])])
  for i in range(len(borders) - 1):
    n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
  n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))
  res_arr = np.divide(np.multiply((n_arr - p_arr * 100), (n_arr - p_arr * 100)), p_arr * 100)
  print(f"n_i: \n" + str(n_arr) + "\n"
        f"Сумма: " + str(np.sum(n_arr)) + "\n"
        f"n_i  - n*p_i: "+str(n_arr - p_arr * sz) + "\n"
        f"res: " + str(res_arr) + "\n"
        f"res_sum = " + str(np.sum(res_arr)) + "\n")


  rows = [[i + 1, ("%.2f" % borders[(i + 1) % (k - 1)]) + ', ' + "%.2f" % borders[(i + 2) % (k - 1)],
                   "%.2f" % n_arr[i],
                   "%.4f" % p_arr[i],
                   "%.2f" % (sz * p_arr[i]),
                   "%.2f" % (n_arr[i] - sz * p_arr[i]),
                   "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
  rows.insert(0, ['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', '$np_i$' '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
  with open("task7_data/task7_normal.txt", "w") as f:
      f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
      f.write("\\hline\n")
      for row in rows:
        f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
        f.write("\\hline\n")
      f.write("\\end{tabular}")


def laplace_exp():
  sz = 20
  samples = np.random.laplace(0, 1 / math.sqrt(2), size=sz)
  mu_c = np.mean(samples)
  sigma_c = np.std(samples)

  alpha = 0.05
  p = 1 - alpha
  k = 7
  left = mu_c - 3
  right = mu_c + 3
  borders = np.linspace(left, right, num=(k - 1))
  p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
  for i in range(len(borders) - 1):
    val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
    p_arr = np.append(p_arr, val)
  p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))
  print(f"mu: " + str(mu_c) + ", sigma: " + str(sigma_c))
  print(f"Промежутки: " + str(borders) + "\n"
        f"p_i: \n " + str(p_arr) + " \n"
        f"n * p_i: \n " + str(p_arr * sz) + "\n"
        f"Сумма: " + str(np.sum(p_arr)) + "\n")
  n_arr = np.array([len(samples[samples <= borders[0]])])
  for i in range(len(borders) - 1):
    n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
  n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))
  res_arr = np.divide(np.multiply((n_arr - p_arr * sz), (n_arr - p_arr * sz)), p_arr * sz)
  print(f"n_i: \n" + str(n_arr) + "\n"
        f"Сумма: " + str(np.sum(n_arr)) + "\n"
        f"n_i  - n*p_i: "+str(n_arr - p_arr * sz) + "\n"
        f"res: " + str(res_arr) + "\n"
        f"res_sum = " + str(np.sum(res_arr)) + "\n")


  rows = [[i + 1, ("%.2f" % borders[(i + 1) % (k - 1)]) + ', ' + "%.2f" % borders[(i + 2) % (k - 1)],
                   "%.2f" % n_arr[i],
                   "%.4f" % p_arr[i],
                   "%.2f" % (sz * p_arr[i]),
                   "%.2f" % (n_arr[i] - sz * p_arr[i]),
                   "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
  rows.insert(0, ['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', '$np_i$' '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
  with open("task7_data/task7_laplace.txt", "w") as f:
      f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
      f.write("\\hline\n")
      for row in rows:
        f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
        f.write("\\hline\n")
      f.write("\\end{tabular}")


def task7():
  if not os.path.exists("task7_data"):
    os.mkdir("task7_data")
  print("Нормальное распределение")
  normal_exp()
  print("Распределение Лапласа")
  laplace_exp()


if __name__ == "__main__":
  task7()
