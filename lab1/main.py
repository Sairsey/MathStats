import numpy as np
import matplotlib.pyplot as plt
from math import factorial, erf, pow, gamma
import csv
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde
from scipy.stats import poisson
import seaborn as sns

import os

num_bins = 20

distr_type = ['Norm', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


def get_distr(d_name, num):
    if d_name == 'Norm':
        return np.random.normal(0, 1, num)
    elif d_name == 'Cauchy':
        return np.random.standard_cauchy(num)
    elif d_name == 'Laplace':
        return np.random.laplace(0, np.sqrt(2) / 2, num)
    elif d_name == 'Poisson':
        return np.random.poisson(10, num)
    elif d_name == 'Uniform':
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), num)
    return []

def get_func(d_name, x):
    if d_name == 'Norm':
        return 0.5 * (1 + erf(x / np.sqrt(2)))
    elif d_name == 'Cauchy':
        return np.arctan(x) / np.pi + 0.5
    elif d_name == 'Laplace':
        if x <= 0:
            return 0.5 * np.exp(np.sqrt(2) * x)
        else:
            return 1 - 0.5 * np.exp(-np.sqrt(2) * x)
    elif d_name == 'Poisson':
        return poisson.cdf(x, 10)
    elif d_name == 'Uniform':
        if x < -np.sqrt(3):
            return 0
        elif np.fabs(x) <= np.sqrt(3):
            return (x + np.sqrt(3)) / (2 * np.sqrt(3))
        else:
            return 1
    return 0

def get_func_array(d_name, array):
  return [get_func(d_name, x) for x in array]

def get_density_func(d_name, array):
    if d_name == 'Norm':
        return [1 / (1 * np.sqrt(2 * np.pi)) * np.exp( - (x - 0) ** 2 / (2 * 1 ** 2) ) for x in array]
    elif d_name == 'Cauchy':
        return [1 / (np.pi * (x ** 2 + 1) ) for x in array]
    elif d_name == 'Laplace':
        return [1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(x)) for x in array]
    elif d_name == 'Poisson':
        return [10 ** x * np.exp(-10) / gamma(x + 1) for x in array]
    elif d_name == 'Uniform':
        return [1 / (2 * np.sqrt(3)) if abs(x) <= np.sqrt(3) else 0 for x in array]
    return []

def cm_to_inch(value):
  return value/2.54

def task1():
  sizes = [10, 50, 1000]
  for distr_name in distr_type:
    fig, ax = plt.subplots(1, 3, figsize=(cm_to_inch(30),cm_to_inch(10)))
    plt.subplots_adjust(wspace=0.5)
    fig.suptitle(distr_name)
    for i in range(len(sizes)):
      array = get_distr(distr_name, sizes[i])
      n, bins, patches = ax[i].hist(array, num_bins, density=1, edgecolor='black', alpha = 0.2)
      ax[i].plot(bins, get_density_func(distr_name, bins), color = 'r', linewidth=1)
      ax[i].set_title("n = " + str(sizes[i]))
    if not os.path.isdir('task1_data'):
      os.makedirs("task1_data")
    plt.savefig("task1_data/" + distr_name + ".png")
  plt.show()


def calc_quart(array, p):
  new_array = np.sort(array)
  k = len(array) * p
  if k.is_integer():
      return new_array[int(k)]
  else:
      return new_array[int(k) + 1]


def calc_z_q(array):
    low_q = 0.25
    high_q = 0.75
    return (calc_quart(array, low_q) + calc_quart(array, high_q)) / 2


def calc_z_tr(array):
    r = int(len(array) * 0.25)
    new_array = np.sort(array)
    sum = 0
    for i in range(r + 1, len(array) - r):
        sum += new_array[i]
    return sum / (len(array) - 2 * r)

def task2():
  number_of_repeats = 1000
  quan_of_numbers = [10, 100, 1000]
  #clean file
  if not os.path.isdir('task2_data'):
    os.makedirs("task2_data")
  with open("task2_data/task2.csv", "w") as f:
    pass

  for dist_name in distr_type:
    i = 0
    rows = []
    headers = [dist_name, "$\\overline{x}$ (\\ref{mean})", "$med x$ (\\ref{med})", "$z_R$ (\\ref{zr})", "$z_Q$ (\\ref{zq})", "$z_{tr}$ (\\ref{tr_mean})"]
    for N in quan_of_numbers:
      mean = []
      mn = []
      mx = []
      med = []
      z_r = []
      z_q_ = []
      z_tr_ = []
      for rep in range(0, number_of_repeats):
        array = get_distr(dist_name, N)
        array_sorted = np.sort(array)
        mean.append(np.mean(array))
        med.append(np.median(array))
        z_r.append((array_sorted[0] + array_sorted[-1]) / 2)
        z_q_.append(calc_z_q(array))
        z_tr_.append(calc_z_tr(array))
        mn.append(array[0])
        mx.append(array[-1])
      rows.append(["n =" + str(N)])
      rows.append(["$E(z)$",
                        np.around(np.mean(mean), decimals=6),
                        np.around(np.mean(med), decimals=6),
                        np.around(np.mean(z_r), decimals=6),
                        np.around(np.mean(z_q_), decimals=6),
                        np.around(np.mean(z_tr_), decimals=6),
                        ])
      rows.append(["$D(z)$",
                   np.around(np.std(mean) * np.std(mean), decimals=6),
                   np.around(np.std(med) * np.std(med), decimals=6),
                   np.around(np.std(z_r) * np.std(z_r), decimals=6),
                   np.around(np.std(z_q_) * np.std(z_q_), decimals=6),
                   np.around(np.std(z_tr_) * np.std(z_tr_), decimals=6)])
      rows.append(["$E(z) - \sqrt{D(z)}$",
                        np.around(np.mean(mean) - np.std(mean), decimals=6),
                        np.around(np.mean(med) - np.std(med), decimals=6),
                        np.around(np.mean(z_r) - np.std(z_r), decimals=6),
                        np.around(np.mean(z_q_) - np.std(z_q_), decimals=6),
                        np.around(np.mean(z_tr_) - np.std(z_tr_), decimals=6)])
      rows.append(["$E(z) + \sqrt{D(z)}$",
                        np.around(np.mean(mean) + np.std(mean), decimals=6),
                        np.around(np.mean(med) + np.std(med), decimals=6),
                        np.around(np.mean(z_r) + np.std(z_r), decimals=6),
                        np.around(np.mean(z_q_) + np.std(z_q_), decimals=6),
                        np.around(np.mean(z_tr_) + np.std(z_tr_), decimals=6)])
      rows.append(["$\hat{E(z)} $",
                        np.around(np.mean(mean), decimals=6),
                        np.around(np.mean(med), decimals=6),
                        np.around(np.mean(z_r), decimals=6),
                        np.around(np.mean(z_q_), decimals=6),
                        np.around(np.mean(z_tr_), decimals=6)])

      #rows.append([" interval" + str(N), 
      #             "[" + str(rows[-2][1] - rows[-1][1]) + ", " + str(rows[-2][1] + rows[-1][1]) + "]",
      #             "[" + str(rows[-2][2] - rows[-1][2]) + ", " + str(rows[-2][2] + rows[-1][2]) + "]",
      #             "[" + str(rows[-2][3] - rows[-1][3]) + ", " + str(rows[-2][3] + rows[-1][3]) + "]",
      #             "[" + str(rows[-2][4] - rows[-1][4]) + ", " + str(rows[-2][4] + rows[-1][4]) + "]",
      #             "[" + str(rows[-2][5] - rows[-1][5]) + ", " + str(rows[-2][5] + rows[-1][5]) + "]"])
      i += 1
    print(rows)
    if not os.path.isdir('task2_data'):
      os.makedirs("task2_data")
    with open("task2_data/task2_" + dist_name + ".txt", "w") as f:
      f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
      f.write("\\hline\n")
      f.write(" & $\\overline{x}$ (\\ref{mean}) & $med x$ (\\ref{med}) & $z_R$ (\\ref{zr}) & $z_Q$ (\\ref{zq}) & $z_{tr}$ (\\ref{tr_mean}) \\\\\n")
      f.write("\\hline\n")
      for row in rows:
        if len(row) == 1:
          f.write(row[0] + " & " * 5 + "\\\\\n")
        else:
          f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
        f.write("\\hline\n")
      f.write("\\end{tabular}")


def task3():
  number_of_repeats = 1000
  quan_of_numbers = [20, 100]
  rows = []
  for dist_name in distr_type:
    array_20 = get_distr(dist_name, quan_of_numbers[0])
    array_100 = get_distr(dist_name, quan_of_numbers[1])

    for el in array_20:
      if el <= -100 or el >= 100:
        array_20 = np.delete(array_20, list(array_20).index(el))

    for el in array_100:
      if el <= -100 or el >= 100:
        array_100 = np.delete(array_100, list(array_100).index(el))

    line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
    bbox_props = dict(color="b", alpha=0.9)
    flier_props = dict(marker="o", markersize=4)
    plt.boxplot((array_20, array_100), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"], vert=False)
    plt.ylabel("c")
    plt.xlabel("X")
    plt.title(dist_name)
    if not os.path.isdir('task3_data'):
      os.makedirs("task3_data")
    plt.savefig("task3_data/" + dist_name + ".png")
    plt.figure()

    count = 0
    
    for j in range(len(quan_of_numbers)):
      for i in range(number_of_repeats):
        array = get_distr(dist_name, quan_of_numbers[j])

        X = []

        X.append(np.quantile(array, 0.25) - 1.5 * (np.quantile(array, 0.75) - np.quantile(array, 0.25)))
        X.append(np.quantile(array, 0.75) + 1.5 * (np.quantile(array, 0.75) - np.quantile(array, 0.25)))

        for k in range(0, quan_of_numbers[j]):
            if array[k] > X[1] or array[k] < X[0]:
                count += 1

      count /= number_of_repeats
      rows.append([dist_name + ", n = " + str(quan_of_numbers[j]),  np.around(count / quan_of_numbers[j], decimals=3)])
  if not os.path.isdir('task3_data'):
      os.makedirs("task3_data")
  with open("task3_data/task3.csv", "w") as f:
    # create the csv writer
    writer = csv.writer(f)
    for row in rows:
      writer.writerow(row)
  plt.show()

def task41():
  quan_of_numbers = [20, 60, 100]
  repeat = 1000
  for dist_name in distr_type:
    if dist_name == "Poisson":
      array_global = np.arange(6, 14, 1)
    else:
      array_global = np.arange(-4, 4, 0.01)
    fig, ax = plt.subplots(1, 3, figsize=(cm_to_inch(30),cm_to_inch(10)))
    plt.subplots_adjust(wspace=0.5)
    fig.suptitle(dist_name)
    for j in range(len(quan_of_numbers)):
      array = get_distr(dist_name, quan_of_numbers[j])

      for el in array:
        if dist_name == "Poisson" and (el < 6 or el > 14):
          array = np.delete(array, list(array).index(el))
        elif dist_name != "Poisson" and (el < -4 or el > 4):
          array = np.delete(array, list(array).index(el))


      ax[j].set_title('n = ' + str(quan_of_numbers[j]))
      
      if dist_name == 'Poisson':
        ax[j].step(array_global, get_func_array(dist_name, array_global), color='blue', linewidth=0.8)
      else:
        ax[j].plot(array_global, get_func_array(dist_name, array_global), color='blue', linewidth=0.8)
      
      if dist_name == 'Poisson':
        array_ex = np.linspace(6, 14)
      else:
        array_ex = np.linspace(-4, 4)
      ecdf = ECDF(array)
      y = ecdf(array_ex)
      ax[j].step(array_ex, y, color='black')
      if not os.path.isdir('task4_data'):
        os.makedirs("task4_data")
      plt.savefig("task4_data/" + dist_name + "_emperic_f.png")
  plt.show()

def task42():
  quan_of_numbers = [20, 60, 100]
  repeat = 1000
  for dist_name in distr_type:
    if dist_name == "Poisson":
      array_global = np.arange(6, 14, 1)
    else:
      array_global = np.arange(-4, 4, 0.01)
    for j in range(len(quan_of_numbers)):
      array = get_distr(dist_name, quan_of_numbers[j])
      
      for el in array:
        if dist_name == "Poisson" and (el < 6 or el > 14):
          array = np.delete(array, list(array).index(el))
        elif dist_name != "Poisson" and (el < -4 or el > 4):
          array = np.delete(array, list(array).index(el))

      tit = ["h = 1/2 * h_n", "h = h_n", "h = 2 * h_n"]
      bw = [0.5, 1, 2]
      fig, ax = plt.subplots(1, 3, figsize=(cm_to_inch(30),cm_to_inch(10)))
      plt.subplots_adjust(wspace=0.5)
      for k in range(len(bw)):
        kde = gaussian_kde(array, bw_method='silverman')
        h_n = kde.factor
        fig.suptitle(dist_name + ', n = ' + str(quan_of_numbers[j]))
        ax[k].plot(array_global, get_density_func(dist_name, array_global), color='blue', alpha=0.5, label='density')
        ax[k].set_title(tit[k])
        sns.kdeplot(array, ax=ax[k], bw=h_n*bw[k], label='kde')
        ax[k].set_xlabel('x')
        ax[k].set_ylabel('f(x)')
        ax[k].set_ylim([0, 1])
        if dist_name == 'Poisson':
          ax[k].set_xlim([6, 14])
        else:
          ax[k].set_xlim([-4, 4])
        ax[k].legend()
      if not os.path.isdir('task4_data'):
        os.makedirs("task4_data")
      plt.savefig("task4_data/" + dist_name + "_kernel_n" + str(quan_of_numbers[j]) + ".png")
    
  plt.show()

if __name__ == "__main__":
  task2()