from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from workspace.Preprocess import process_data
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA
import argparse, glob
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

n_components = 8


def ica_feature(train, valid):
    len_train = train.shape[0]
    data = np.concatenate((train, valid))
    ica = FastICA(n_components=n_components)
    ica.fit(data.T)
    w = ica.components_.T
    w_min, w_max = np.min(w), np.max(w)
    w = (w - w_min) / (w_max - w_min)
    return w[:len_train], w[len_train:]


def ica_feature_test(train, test):
    len_train = train.shape[0]
    lens = [t.shape[0] for t in test]
    test = np.vstack([t for t in test])
    data = np.concatenate((train, test))
    ica = FastICA(n_components=n_components)
    ica.fit(data.T)
    w = ica.components_.T
    w_min, w_max = np.min(w), np.max(w)
    w = (w - w_min) / (w_max - w_min)
    train, test = w[:len_train], w[len_train:]
    tmp, now = [], 0
    for l in lens:
        tmp.append(test[now: now + l])
        now += l
    return train, np.array(tmp, dtype=object)


def cross_valid():
    accuracies_cv = {}
    for model in (
            svm.SVC(kernel='rbf', gamma=81, C=1),
            KNeighborsClassifier(4, weights='distance'),
            DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    ):
        accuracies = []
        for fold_index in range(min(process_data.FOLD, 10)):
            X_valid, y_valid = dataset[fold_index]
            X_valid = X_valid.astype('float32')
            y_valid = y_valid.astype('float32')
            train_data = np.concatenate((dataset[: fold_index], dataset[fold_index + 1:]))
            X_train = np.vstack(list(train_data[:, 0])).astype('float32')
            y_train = np.hstack(list(train_data[:, 1])).astype('float32')
            X_train, X_valid = ica_feature(X_train, X_valid)
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            accuracies.append(accuracy_score(y_valid, predictions))
        accuracies_cv[model.__repr__()] = np.mean(accuracies)
    print(accuracies_cv)


def _test(method):
    from collections import Counter
    X_train = np.vstack(list(dataset[:, 0])).astype('float32')
    y_train = np.hstack(list(dataset[:, 1])).astype('float32')
    X_test = np.load("workspace/data/test.npy", allow_pickle=True)
    for i in range(X_test.shape[0]):
        sample_X = X_test[i].astype('float32')
        for j in range(sample_X.shape[0]):
            X_min = np.min(sample_X[j])
            X_test[i][j] = (sample_X[j] - X_min) / (np.max(sample_X[j]) - X_min)
    X_train, X_test = ica_feature_test(X_train, X_test)
    model = {
        "svm": svm.SVC(kernel='rbf', gamma=81, C=1),
        "knn": KNeighborsClassifier(4, weights='distance'),
        "dt": DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    }[method]
    model.fit(X_train, y_train)
    submission = open("workspace/submission.csv", "w")
    submission.write("id,categories\r\n")
    for i, data_path in enumerate(glob.glob(process_data.TEST_PATH + "/*")):
        sample_i = data_path.split("/")[-1].split(".")[0]
        predict = model.predict(X_test[i])
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
    parser.add_argument('--method', type=str, default='knn', help='test methods: svm, knn, dt')
    dict_args = vars(parser.parse_args())
    if dict_args['test']:
        _test(method=dict_args['method'])
    else:
        cross_valid()
