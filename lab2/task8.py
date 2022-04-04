import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2, t, norm, moment
import scipy.stats as stats

gamma = 0.95

alpha = 1 - gamma

def get_student_mo(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    t_a = t.ppf(1 - alpha / 2, n - 1)
    q_1 = med - s * t_a / np.sqrt(n - 1)
    q_2 = med + s * t_a / np.sqrt(n - 1)
    return q_1, q_2

def get_chi_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    q_1 =  s * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    q_2 = s * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
    return q_1, q_2


def get_as_mo(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    q_1 = med - s * u / np.sqrt(n)
    q_2 = med + s * u / np.sqrt(n)
    return q_1, q_2


def get_as_sigma(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    m4 = moment(samples, 4)
    e = m4 / (s * s * s * s)
    U = u * np.sqrt((e + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2


def task8():
  samples20 = np.random.normal(0, 1, size=20)
  samples100 = np.random.normal(0, 1, size=100)

  # classic interval params
  student_20 = get_student_mo(samples20, alpha)
  student_100 = get_student_mo(samples100, alpha)

  chi_20 = get_chi_sigma(samples20, alpha)
  chi_100 = get_chi_sigma(samples100, alpha)

  # assimptotic interval params
  as_mo_20 = get_as_mo(samples20, alpha)
  as_mo_100 = get_as_mo(samples100, alpha)

  as_d_20 = get_as_sigma(samples20, alpha)
  as_d_100 = get_as_sigma(samples100, alpha)

  print(f"Асимптотический подход:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")

  print(f"Классический подход:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(student_100) + " \t sigma: " + str(chi_100) + "\n")



if __name__ == "__main__":
  task8()
