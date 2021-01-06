from sklearn.tree import DecisionTreeClassifier
from workspace.Preprocess import process_data
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import glob

# ignore ConvergenceWarning
import warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# load data
dataset = np.array([
    [np.load(f"../data/ica_train_{i}.npy", allow_pickle=True), np.load(f"../data/ica_label_{i}.npy")]
    for i in range(process_data.FOLD)
], dtype=object)

# normalization
for i in range(process_data.FOLD):
    X_data = dataset[i, 0]
    for j in range(X_data.shape[0]):
        for k in range(X_data[j].shape[0]):
            X_min = np.min(X_data[j][k])
            X_data[j][k] = (X_data[j][k] - X_min) / (np.max(X_data[j][k]) - X_min)


def ica_feature(data):
    features, labels = [], []
    n_components = 12
    for i in range(data.shape[0]):
        ica = FastICA(n_components=n_components)
        if data[i].shape[0] < n_components:
            data[i] = np.repeat(data[i], n_components // data[i].shape[0] + 1, axis=0)
        # ica.fit(data[i].T)
        # w = ica.components_
        # u = np.dot(w, data[i]).T
        u = ica.fit_transform(data[i].T).T
        # z = u.flatten()
        # plt.plot(np.arange(z.shape[0]), z)
        # plt.show()
        features.append(u.flatten())
    return np.vstack(features)


def cross_valid():
    for model in (DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3),):
        accuracies = []
        for fold_index in range(min(process_data.FOLD, 1)):
            X_valid, y_valid = dataset[fold_index]
            y_valid = y_valid.astype("float")
            train_data = np.concatenate((dataset[: fold_index], dataset[fold_index + 1:]))
            X_train = np.hstack(list(train_data[:, 0]))
            y_train = np.hstack(list(train_data[:, 1])).astype("float")
            X_train = ica_feature(X_train)
            model.fit(X_train, y_train)
            predictions = model.predict(ica_feature(X_valid))
            accuracy = accuracy_score(y_valid, predictions)
            accuracies.append(accuracy)
        print(model, accuracies)


def _test():
    from collections import Counter
    X_train = np.hstack(list(dataset[:, 0]))
    y_train = np.hstack(list(dataset[:, 1])).astype('float32')
    X_test = np.load("../data/test.npy", allow_pickle=True)
    for i in range(X_test.shape[0]):
        sample_X = X_test[i].astype('float32')
        for j in range(sample_X.shape[0]):
            X_min = np.min(sample_X[j])
            X_test[i][j] = (sample_X[j] - X_min) / (np.max(sample_X[j]) - X_min)
    X_test = ica_feature(X_test)
    model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    model.fit(ica_feature(X_train), y_train)
    submission = open("../submission.csv", "w")
    submission.write("id,categories\r\n")
    prediction = model.predict(X_test).astype("int32")
    for i, data_path in enumerate(glob.glob(process_data.TEST_PATH + "/*")):
        sample_i = data_path.split("/")[-1].split(".")[0]
        submission.write(f"{sample_i},{prediction[i]}\r\n")
        print(f"{sample_i:>7}, {prediction[i]}")
    submission.close()


if __name__ == "__main__":
    # _test()
    cross_valid()
