from sklearn.neighbors import KNeighborsClassifier
from workspace.Preprocess import process_data
from sklearn.metrics import accuracy_score
from scipy.fftpack import fft
from scipy import interpolate
import glob, argparse
import numpy as np

# ignore FutureWarning
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

# load dataset
dataset = np.array([
    [np.load(f"workspace/data/train_{i}.npy"), np.load(f"workspace/data/label_{i}.npy")]
    for i in range(process_data.FOLD)
], dtype=object)

# normalization
for i in range(process_data.FOLD):
    X_data = dataset[i, 0]
    for j in range(X_data.shape[0]):
        X_min = np.min(X_data[j])
        X_data[j] = (X_data[j] - X_min) / (np.max(X_data[j]) - X_min)


def fft_feature(data):
    features = []
    for i in range(data.shape[0]):
        time_feature = data[i, ::2]
        fft_y = fft(time_feature)
        freq_feature = np.abs(fft_y) / 200
        freq_feature = freq_feature[1:100]
        f = interpolate.interp1d(np.linspace(0, 1, freq_feature.shape[0]), freq_feature)
        freq_feature = f(np.linspace(0, 1, time_feature.shape[0])) * 12.0  # 5 is scaling coefficient
        freq_feature = np.concatenate((time_feature, freq_feature))
        features.append(freq_feature)
        # plt.plot(np.arange(freq_feature.shape[0]), freq_feature)
        # plt.show()
    return np.vstack(features)


def cross_valid():
    accuracies_cv = []
    for model in (KNeighborsClassifier(k, 'uniform') for k in np.arange(11, 12)):
        accuracies = []
        for fold_index in range(min(process_data.FOLD, 10)):
            X_valid, y_valid = dataset[fold_index]
            X_valid = X_valid.astype('float32')
            y_valid = y_valid.astype('float32')
            train_data = np.concatenate((dataset[: fold_index], dataset[fold_index + 1:]))
            X_train = np.vstack(list(train_data[:, 0])).astype('float32')
            y_train = np.hstack(list(train_data[:, 1])).astype('float32')
            model.fit(fft_feature(X_train), y_train)
            predictions = model.predict(fft_feature(X_valid))
            accuracies.append(accuracy_score(y_valid, predictions))
        accuracies_cv.append(np.mean(accuracies))
    print(accuracies_cv)


def _test():
    from collections import Counter
    X_train = np.vstack(list(dataset[:, 0])).astype('float32')
    y_train = np.hstack(list(dataset[:, 1])).astype('float32')
    X_test = np.load("workspace/data/test.npy", allow_pickle=True)
    model = KNeighborsClassifier(4, 'distance')
    model.fit(fft_feature(X_train), y_train)
    submission = open("workspace/submission.csv", "w")
    submission.write("id,categories\r\n")
    for i, data_path in enumerate(glob.glob(process_data.TEST_PATH + "/*")):
        sample_i = data_path.split("/")[-1].split(".")[0]
        sample_X = X_test[i].astype('float32')
        for j in range(sample_X.shape[0]):
            X_min = np.min(sample_X[j])
            sample_X[j] = (sample_X[j] - X_min) / (np.max(sample_X[j]) - X_min)
        predict = model.predict(fft_feature(sample_X))
        predict = int(max(Counter(predict).items(), key=lambda x: x[1])[0])
        submission.write(f"{sample_i},{predict}\r\n")
        print(f"{sample_i:>7}, {predict}")
    submission.close()


def boolean_flag(parser, name, default=False, help=None):
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    boolean_flag(parser, 'test', default=False)
    dict_args = vars(parser.parse_args())
    if dict_args['test']:
        _test()
    else:
        cross_valid()
