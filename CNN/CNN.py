from tensorflow import keras
from tensorflow.python.keras.layers import (
    Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D
)
from workspace.Preprocess import process_data
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_file = 'model/cnn_{}.h5'

dataset = np.array([
    [np.load(f"../data/train_{i}.npy"), np.load(f"../data/label_{i}.npy")]
    for i in range(process_data.FOLD)
], dtype=object)

accuracies = []
for fold_index in range(process_data.FOLD):

    # load dataset
    X_valid, y_valid = dataset[fold_index]
    X_valid = X_valid.astype('float32')
    y_valid = y_valid.astype('float32')
    train_data = np.concatenate((dataset[: fold_index], dataset[fold_index + 1:]))
    X_train = np.vstack(list(train_data[:, 0])).astype('float32')
    y_train = np.hstack(list(train_data[:, 1])).astype('float32')
    l_train = X_train.shape[0]
    l_valid = X_valid.shape[0]
    
    # normalization
    for i in range(l_train):
        X_min = np.min(X_train[i])
        X_train[i] = (X_train[i] - X_min) / (np.max(X_train[i]) - X_min)
    for i in range(l_valid):
        X_min = np.min(X_valid[i])
        X_valid[i] = (X_valid[i] - X_min) / (np.max(X_valid[i]) - X_min)

    # reshape
    X_train = X_train.reshape(-1, 1200, 1)
    X_valid = X_valid.reshape(-1, 1200, 1)

    # model definition
    model = keras.models.Sequential()
    model.add(Conv1D(16, 30, activation='relu', input_shape=(1200, 1), padding="same"))
    model.add(MaxPool1D(pool_size=3, strides=3))
    model.add(Conv1D(32, 20, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=3, strides=3))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1, validation_data=(X_valid, y_valid), callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/fold_{fold_index}/', histogram_freq=0, write_graph=True, write_images=False,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
        )
    ])

    model.save(model_file.format(fold_index))
    accuracies.append(history.history["val_accuracy"][-1])
print(accuracies)
