from workspace.Preprocess import process_data
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np

# load dataset
dataset = np.array([
    [np.load(f"../data/train_{i}.npy"), np.load(f"../data/label_{i}.npy")]
    for i in range(process_data.FOLD)
], dtype=object)

# normalization
for i in range(process_data.FOLD):
    X_data = dataset[i, 0]
    for j in range(X_data.shape[0]):
        X_min = np.min(X_data[i])
        X_data[i] = (X_data[i] - X_min) / (np.max(X_data[i]) - X_min)

# build models
C = 1.0  # SVM regulation parameter
models = (svm.SVC(kernel='linear', C=C),  # 'SVC with linear kernel'
          svm.LinearSVC(C=C),  # 'LinearSVC (linear kernel)'
          svm.SVC(kernel='rbf', gamma=0.1, C=C),  # 'SVC with RBF kernel & gamma=0.1'
          svm.SVC(kernel='rbf', gamma=1, C=C),  # 'SVC with RBF kernel & gamma=1'
          svm.SVC(kernel='rbf', gamma=10, C=C),  # 'SVC with RBF kernel & gamma=10'
          svm.SVC(kernel='poly', degree=3, C=C))  # 'SVC with polynomial (degree 3) kernel'

# for fold_index in range(min(process_data.FOLD, 1)):
#
#     # load dataset
#     X_valid, y_valid = dataset[fold_index]
#     X_valid = X_valid.astype('float32')
#     y_valid = y_valid.astype('float32')
#     train_data = np.concatenate((dataset[: fold_index], dataset[fold_index + 1:]))
#     X_train = np.vstack(list(train_data[:, 0])).astype('float32')
#     y_train = np.hstack(list(train_data[:, 1])).astype('float32')
#     l_train = X_train.shape[0]
#     l_valid = X_valid.shape[0]
#
#     # normalization
#     for i in range(l_train):
#         X_min = np.min(X_train[i])
#         X_train[i] = (X_train[i] - X_min) / (np.max(X_train[i]) - X_min)
#     for i in range(l_valid):
#         X_min = np.min(X_valid[i])
#         X_valid[i] = (X_valid[i] - X_min) / (np.max(X_valid[i]) - X_min)
#
#
#
#     model = svm.SVC(kernel='rbf', gamma=0.1, C=C)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_valid)
#     accuracy = accuracy_score(y_valid, predictions)
#     print(accuracy)
