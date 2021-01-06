# -*- coding: utf-8
import matplotlib.pyplot as plt
from functools import reduce
from biosppy.signals import ecg
import scipy.io as io
import numpy as np
import glob, copy, pywt, argparse

TRAIN_PATH = "workspace/ECGTrainData/Train/"
TEST_PATH = "workspace/ECGTestData/ECGTestData/"
DAO_TOTAL = 12
FOLD = 5
CATE_MAP = {
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


def concatenate():
    train, test = {"inputs": [], "labels": [], "size": []}, []
    for cate_path in glob.glob(TRAIN_PATH + "/*"):
        for data_path in glob.glob(cate_path + "/*"):
            data = io.loadmat(data_path)['Beats']
            [[size]], [label], [_], [sample] = data[0][0]
            data_con = np.array(reduce(lambda x, y: np.concatenate((x, y)), sample))
            for dao_id in range(DAO_TOTAL):
                bound = data_con[:, dao_id].shape[0]
                compr = data_con[:, dao_id][: bound // 8]
                if np.abs(np.max(compr)) < np.abs(np.min(compr)):
                    data_con[:, dao_id] = -data_con[:, dao_id]
            train["inputs"].append(data_con[:, :][::2])
            train["labels"].append(CATE_MAP[label])
            train["size"].append(size)
    for data_path in glob.glob(TEST_PATH + "/*"):
        data = io.loadmat(data_path)['data']
        for dao_id in range(1, DAO_TOTAL + 1):
            bound = data[:, dao_id].shape[0]
            compr = data[:, dao_id][: bound // 8]
            if np.abs(np.max(compr)) < np.abs(np.min(compr)):
                data[:, dao_id] = -data[:, dao_id]
        test.append(data[:, 1:])
    return train, test


def segmentation(data, label=None):
    inputs, labels = [], []
    for sap_id in range(len(data)):
        data_ref = denoise(copy.copy(data[sap_id][:, 9]))
        r_peaks = ecg.hamilton_segmenter(data_ref, 500).as_dict()['rpeaks']
        beats = np.hstack([ecg.extract_heartbeats(
            signal=denoise(copy.copy(data[sap_id][:, dao_id])),
            rpeaks=r_peaks, sampling_rate=500, before=0.2, after=0.4
        )["templates"] for dao_id in [1, 3, 6, 9]])
        inputs.append(beats)
        if label: labels.append(np.full(beats.shape[0], label[sap_id]))
        # if label: labels.append(label[sap_id])
    if not label:
        return np.array(inputs, dtype=object)
    else:
        return np.vstack(inputs), np.hstack(labels)
        # return np.array(inputs), np.array(labels)


def denoise(data):
    w = pywt.Wavelet('db8')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], 0.3 * max(coeffs[i]))
    return pywt.waverec(coeffs, 'db8')


def display(data, sample_id, dao_id):
    plt.figure(figsize=(16, 9))
    data_sav = data[sample_id][:, dao_id]
    plt.plot(np.arange(data_sav.shape[0]), data_sav)
    plt.show()


def boolean_flag(parser, name, default=False, help=None):
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    boolean_flag(parser, 'ica', default=False)
    dict_args = vars(parser.parse_args())
    if dict_args['ica']:
        train, test = concatenate()
        xs, ys = segmentation(train["inputs"], train["labels"])
        random_index = np.arange(xs.shape[0])
        np.random.shuffle(random_index)
        each_fold = xs.shape[0] // FOLD
        for i in range(FOLD - 1):
            index = random_index[i * each_fold: (i + 1) * each_fold]
            np.save(f'workspace/data/ica_train_{i}.npy', xs[index])
            np.save(f'workspace/data/ica_label_{i}.npy', ys[index])
        index = random_index[(FOLD - 1) * each_fold:]
        np.save(f'workspace/data/ica_train_{FOLD - 1}.npy', xs[index])
        np.save(f'workspace/data/ica_label_{FOLD - 1}.npy', ys[index])
    else:
        train, test = concatenate()
        xs, ys = segmentation(train["inputs"], train["labels"])
        random_index = np.arange(xs.shape[0])
        np.random.shuffle(random_index)
        each_fold = xs.shape[0] // FOLD
        for i in range(FOLD - 1):
            index = random_index[i * each_fold: (i + 1) * each_fold]
            np.save(f'workspace/data/train_{i}.npy', xs[index])
            np.save(f'workspace/data/label_{i}.npy', ys[index])
        index = random_index[(FOLD - 1) * each_fold:]
        np.save(f'workspace/data/train_{FOLD - 1}.npy', xs[index])
        np.save(f'workspace/data/label_{FOLD - 1}.npy', ys[index])
        np.save('workspace/data/test.npy', segmentation(test))
