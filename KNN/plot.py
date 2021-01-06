import matplotlib.pyplot as plt
import numpy as np

accuracies_uniform = [0.9127300025555838, 0.8769620495783286, 0.8934142601584462, 0.879465882954255, 0.8812586250958343, 0.8826878354203936, 0.874456299514439, 0.8669499105545618, 0.8673044978277537, 0.8562158190646564]
accuracies_distance = [0.9127300025555838, 0.9127300025555838, 0.9041464349603885, 0.9048600817786865, 0.8959232047022743, 0.8987835420393561, 0.8909136212624584, 0.8869792997699975, 0.8812522361359572, 0.8712394582162023]
plt.plot(np.arange(1, 11), accuracies_uniform, marker='s', label="Weights=Uniform")
plt.plot(np.arange(1, 11), accuracies_distance, marker='o', label="Weights=Distance")
plt.xticks(np.arange(1, 11))
plt.yticks(np.around(np.linspace(0.85, 0.92, 10), 2))
plt.grid()
plt.legend(loc="lower left")
plt.xlabel("K values")
plt.ylabel("Evaluation Accuracy")
plt.title("Evaluation Accuracy of K-NN models with Different K Values")
plt.show()
