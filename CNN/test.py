from tensorflow.keras.models import load_model
from workspace.Preprocess import process_data
import numpy as np
import glob

# model = load_model('workspace/CNN/log_sav/optimal/cnn_2.h5') # CNN method version 1.0
model = load_model('workspace/CNN/log_sav/optimal_30_20/cnn_0.h5')
X_test = np.load("workspace/data/test.npy", allow_pickle=True)
submission = open("workspace/submission.csv", "w")
submission.write("id,categories\r\n")
for i, data_path in enumerate(glob.glob(process_data.TEST_PATH + "/*")):
    sample_i = data_path.split("/")[-1].split(".")[0]
    sample_X = X_test[i]
    for j in range(sample_X.shape[0]):
        X_min = np.min(sample_X[j])
        sample_X[j] = (sample_X[j] - X_min) / (np.max(sample_X[j]) - X_min)
    sample_X = sample_X.reshape(-1, 1200, 1)
    predict = model.predict(sample_X)
    predict = np.argmax(np.sum(predict, axis=0))
    submission.write(f"{sample_i},{predict}\r\n")
    print(f"{sample_i:>7}, {predict}")
submission.close()
