import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use('./deeplearning.mplstyle')

filepath = "./data.txt"
x_train = []
y_train = []
with open(filepath, 'r') as file:
    for line in file:
        data = line.strip().split(',')
        x_train.append(int(data[0]))
        y_train.append(int(data[2]))

x_train = np.array(x_train)
y_train = np.array(y_train)

# number of training examples
m = len(x_train)

def compute_cost(x, y, w, b):
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# plt_intuition(x_train, y_train)
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()
