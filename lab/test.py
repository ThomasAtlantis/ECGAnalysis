# -*- coding: utf-8
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import glob

cate_map = {
    '窦性心律_左室肥厚伴劳损': 0,
    '窦性心动过缓': 1,
    '窦性心动过速': 2,
    '窦性心律_不完全性右束支传导阻滞': 3,
    '窦性心律_电轴左偏': 4,
    '窦性心律_提示左室肥厚': 5,
    '窦性心律_完全性右束支传导阻滞': 6,
    '窦性心律_完全性左束支传导阻滞': 7,
    '窦性心律_左前分支阻滞': 8,
    '正常心电图': 9
}

file_path = "ECGTrainData/Train/窦性心动过缓"
for mat_path in glob.glob(file_path + "/*")[: 1]:
    data = io.loadmat(mat_path)['Beats']
    [[size]], [label], [_], [sample] = data[0][0]
    beat = sample[0]
    for i in range(min(beat.shape[1], 1)):
        plt.plot(np.linspace(0, 1, beat.shape[0]), beat[:, i])
    plt.show()
