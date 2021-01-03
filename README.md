## ECG Analysis
### CNN method

#### Configuration

```shell
conda create -n python38 python==3.8.3
conda install tensorflow=2.4
conda install matplotlib
```

#### How to run

Fill directory `ECGTrainData`, `ECGTestData` with unzipped raw dataset, then run commands below to generate 5 fold numpy binary data files.

```sh
cd ./Preprocess && python process_data.py
```
Train and test CNN model with
```shell
cd ./CNN && python CNN.py
```
models will be stored in `./model/`, logs will be saved in `./logs/`. You can monitor the training process using tensorboard
```shell
tensorboard --log_dir=./logs
```