import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./deeplearning.mplstyle")

filepath = "./data.txt"
x_train = []
y_train = []
with open(filepath, 'r') as file:
    for line in file:
        data = line.strip().split(',')
        x_train.append(int(data[0]))
        y_train.append(int(data[2]))

# print(f"x_train = {x_train}")
# print(f"y_train = {y_train}")

# print(f"x_train shape: {x_train.shape}")
# m = x_train.shape[0]
m = len(x_train)


# print(f"Number of training examples is : {m}")

# for i in range(m):
#     x_i = x_train[i]
#     y_i = y_train[i]
#     print(f"(x^({i}), y^({i})) = ({x_i},{y_i})")

# plt.scatter(x_train, y_train,marker='x',c='r')
# plt.title("Houses Price")
# plt.xlabel("Size (sqft)")
# plt.ylabel("Price (dollars)")
# # plt.show()

w = 150
b = 100

def compute_model_output(x, w, b):
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b, )
plt.plot(x_train, tmp_f_wb, c='b', label="Our Prediction")
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Houses Price")
plt.xlabel("Size (sqft)")
plt.ylabel("Price (dollars)")
plt.show()