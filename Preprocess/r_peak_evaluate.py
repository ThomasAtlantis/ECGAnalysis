import matplotlib.pyplot as plt
import numpy as np


def error_rate(file_name):
    errors = []
    with open(file_name, "r") as reader:
        for line in reader.readlines():
            error_line = [int(x) ** 2 for x in line.strip().split()]
            errors.append(error_line)
    errors = np.array(errors)
    print(file_name, np.sum(errors, axis=0) / 140)
    plt.plot(np.arange(12), np.sum(errors, axis=0) / 140, marker="s", label=file_name.split(".")[0])


error_rate("christov_643.txt")
error_rate("hamilton_19.txt")
error_rate("naive_7.txt")
plt.xlabel("Lead Number")
plt.ylabel("Mean Square Error")
plt.xticks(np.arange(12))
plt.title("MSE of R-Peaks Recognition Algorithms on 12-Leads ECG Signals")
plt.legend(loc="upper right")
plt.show()
