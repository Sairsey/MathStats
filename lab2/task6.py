import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import os

def get_least_squares(x, y):
  xy_med = np.mean(np.multiply(x, y))
  x_med = np.mean(x)
  x_2_med = np.mean(np.multiply(x, x))
  y_med = np.mean(y)
  b1_mnk = (xy_med - x_med * y_med) / (x_2_med - x_med * x_med)
  b0_mnk = y_med - x_med * b1_mnk

  dev = 0
  for i in range(len(x)):
      dev += (b0_mnk + b1_mnk * x[i] - y[i]) ** 2
  print('Невязка МНК:'  + str(math.sqrt(dev)))
  return b0_mnk, b1_mnk

def abs_dev_val(b_arr, x, y):
  return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))

def get_linear_approx(x, y):
  init_b = np.array([0, 1])
  res = minimize(abs_dev_val, init_b, args=(x, y), method='COBYLA')

  dev = 0
  for i in range(len(x)):
    dev += math.fabs(res.x[0] + res.x[1] * x[i] - y[i])
  print('Невязка МНМ:' + str(dev))
  return res.x

def draw(lsm_0, lsm_1, lam_0, lam_1, x, y, title, fname):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', s=6, label='Выборка')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='blue', label='МНК')
    ax.plot(x, y_lam, color='red', label='МНМ')
    ax.plot(x, y_real, color='green', label='Модель')
    ax.set(xlabel='X', ylabel='Y',
       title=title)
    ax.legend()
    ax.grid()
    fig.savefig(fname + '.png', dpi=200)

def task6():
  # values
  x = np.arange(-1.8, 2.1, 0.2)
  # error
  eps = np.random.normal(0, 1, size=20)

  # y = 2 + 2x + eps
  y = np.add(np.add(np.full(20, 2), x * 2), eps)
  # y2 same but with noise
  y2 = np.add(np.add(np.full(20, 2), x * 2), eps)
  y2[0] += 10
  y2[19] += -10

  lsm_0, lsm_1 = get_least_squares(x, y)
  print("МНК, без возмущений: " + str(lsm_0) + ", " + str(lsm_1))
  lam_0, lam_1 = get_linear_approx(x, y)
  print("МНМ, без возмущений: " + str(lam_0) + ", " + str(lam_1))

  lsm_02, lsm_12 = get_least_squares(x, y2)
  print("МНК, с возмущенями: " + str(lsm_02) + ", " + str(lsm_12))
  lam_02, lam_12 = get_linear_approx(x, y2)
  print("МНМ, с возмущенями: " + str(lam_02) + ", " + str(lam_12))

  if not os.path.exists("task6_data"):
    os.mkdir("task6_data")

  draw(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Выборка без возмущений', 'task6_data/no_dev')
  draw(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Выборка с возмущенями', 'task6_data/dev')
  plt.show()

if __name__ == "__main__":
  task6()
