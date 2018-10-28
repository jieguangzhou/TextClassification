## Text Classification

​	基于神经网络的文本分类模型实现包括TextRNN，TextCNN和fasttext。

### 环境

+ python3
+ tensorflow1.9
+ numpy

### 数据集合

​	感谢[gaussic](https://github.com/gaussic/text-classification-cnn-rnn)提供的关于THUCNews的一个子集。

### 可选参数

参数文件可见[text_classification/opt.py](https://github.com/jieguangzhou/TextClassification/blob/master/text_classification/opt.py)

```shell
nn:
  -model MODEL          use model, fasttext, textrnn or textcnn
  -embedding_size EMBEDDING_SIZE
                        embedding size
  -vocab_size VOCAB_SIZE
                        vocab size
  -embedding_path EMBEDDING_PATH
                        embedding path, 暂不使用
  -keep_drop_prob KEEP_DROP_PROB
                        keep_drop_prob
  -class_num CLASS_NUM  class_num

rnn:
  -num_units NUM_UNITS  rnn cell hidden size
  -layer_num LAYER_NUM  rnn layer number
  -cell_type CELL_TYPE  rnn cell type, gru or lstm
  -bidirectional        use bidirectional

cnn:
  -filter_num FILTER_NUM
                        cnn filter num
  -kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]
                        cnn kernel_sizes, a list of int

train:
  -learning_rate LEARNING_RATE
                        learning_rate
  -batch_size BATCH_SIZE
                        batch_size
  -epoch_num EPOCH_NUM
  -print_every_step PRINT_EVERY_STEP
  -save_path SAVE_PATH

data:
  -train_data TRAIN_DATA
                        train data path
  -val_data VAL_DATA    val data path
  -test_data TEST_DATA  test data path
  -vocab_path VOCAB_PATH
                        vocab_pathe
  -label_path LABEL_PATH
                        label_path
  -cut_length CUT_LENGTH
                        cut_length
  -reverse              reverse the sequence

```



### 训练与验证

#### 1 FastText

```shell
python3.5 -m "text_classification.main" -model fasttext -save_path save/fasttext -epoch_num 5
```

```shell
create model

 --------------------
FastText : parms
Fasttext/embedding:0 (5000, 128)
Fasttext/dense/kernel:0 (128, 10)
Fasttext/dense/bias:0 (10,)
-------------------- 

load data set
Epoch: 1
step:   100, train loss:   2.2, train accuary: 42.23%, val loss :   2.2, val accuary: 34.26%, cost:0:00:01.963413
step:   200, train loss:   1.9, train accuary: 50.22%, val loss :   2.0, val accuary: 48.36%, cost:0:00:03.807075
step:   300, train loss:   1.6, train accuary: 67.36%, val loss :   1.7, val accuary: 59.76%, cost:0:00:05.653884
step:   400, train loss:   1.3, train accuary: 75.27%, val loss :   1.5, val accuary: 65.62%, cost:0:00:07.465123
step:   500, train loss:   1.0, train accuary: 79.73%, val loss :   1.3, val accuary: 68.18%, cost:0:00:09.304550
step:   600, train loss:  0.85, train accuary: 81.56%, val loss :   1.2, val accuary: 71.32%, cost:0:00:11.191120
step:   700, train loss:  0.74, train accuary: 83.41%, val loss :   1.0, val accuary: 72.88%, cost:0:00:13.064502
Epoch: 2
step:   800, train loss:  0.63, train accuary: 85.42%, val loss :  0.91, val accuary: 74.66%, cost:0:00:00.761135
step:   900, train loss:  0.58, train accuary: 85.97%, val loss :  0.83, val accuary: 75.72%, cost:0:00:02.631185
step:  1000, train loss:  0.54, train accuary: 86.86%, val loss :  0.76, val accuary: 76.76%, cost:0:00:04.526994
step:  1100, train loss:  0.48, train accuary: 87.98%, val loss :  0.69, val accuary: 78.82%, cost:0:00:06.391131
step:  1200, train loss:  0.48, train accuary: 88.19%, val loss :  0.64, val accuary: 80.36%, cost:0:00:08.226393
step:  1300, train loss:  0.42, train accuary: 89.55%, val loss :  0.59, val accuary: 82.72%, cost:0:00:10.066286
step:  1400, train loss:  0.39, train accuary: 90.25%, val loss :  0.55, val accuary: 84.10%, cost:0:00:11.896519
step:  1500, train loss:  0.37, train accuary: 90.34%, val loss :  0.53, val accuary: 84.36%, cost:0:00:13.723557
Epoch: 3
step:  1600, train loss:  0.36, train accuary: 90.36%, val loss :   0.5, val accuary: 85.36%, cost:0:00:00.998591
step:  1700, train loss:  0.34, train accuary: 91.38%, val loss :  0.47, val accuary: 86.24%, cost:0:00:02.844450
step:  1800, train loss:  0.32, train accuary: 91.50%, val loss :  0.46, val accuary: 86.76%, cost:0:00:04.688984
step:  1900, train loss:  0.33, train accuary: 91.97%, val loss :  0.45, val accuary: 87.16%, cost:0:00:06.528917
step:  2000, train loss:  0.29, train accuary: 92.62%, val loss :  0.43, val accuary: 87.98%, cost:0:00:08.379237
step:  2100, train loss:   0.3, train accuary: 92.39%, val loss :  0.41, val accuary: 88.14%, cost:0:00:10.242890
step:  2200, train loss:  0.29, train accuary: 92.28%, val loss :   0.4, val accuary: 88.18%, cost:0:00:12.107764
step:  2300, train loss:  0.29, train accuary: 92.44%, val loss :  0.38, val accuary: 89.04%, cost:0:00:13.989467
Epoch: 4
step:  2400, train loss:  0.27, train accuary: 93.43%, val loss :  0.38, val accuary: 88.54%, cost:0:00:01.251994
step:  2500, train loss:  0.26, train accuary: 93.34%, val loss :  0.37, val accuary: 88.94%, cost:0:00:03.085236
step:  2600, train loss:  0.24, train accuary: 93.69%, val loss :  0.36, val accuary: 89.62%, cost:0:00:04.896725
step:  2700, train loss:  0.25, train accuary: 93.77%, val loss :  0.35, val accuary: 89.74%, cost:0:00:06.758444
step:  2800, train loss:  0.25, train accuary: 93.47%, val loss :  0.35, val accuary: 89.76%, cost:0:00:08.586772
step:  2900, train loss:  0.25, train accuary: 93.39%, val loss :  0.34, val accuary: 90.62%, cost:0:00:10.414334
step:  3000, train loss:  0.25, train accuary: 93.77%, val loss :  0.34, val accuary: 90.22%, cost:0:00:12.246485
step:  3100, train loss:  0.23, train accuary: 93.84%, val loss :  0.33, val accuary: 90.32%, cost:0:00:14.072625
Epoch: 5
step:  3200, train loss:  0.23, train accuary: 94.27%, val loss :  0.33, val accuary: 90.54%, cost:0:00:01.473536
step:  3300, train loss:  0.22, train accuary: 94.28%, val loss :  0.32, val accuary: 90.82%, cost:0:00:03.316607
step:  3400, train loss:  0.21, train accuary: 94.50%, val loss :  0.31, val accuary: 91.04%, cost:0:00:05.151259
step:  3500, train loss:  0.21, train accuary: 94.48%, val loss :  0.31, val accuary: 91.08%, cost:0:00:07.003222
step:  3600, train loss:  0.22, train accuary: 94.28%, val loss :  0.31, val accuary: 91.22%, cost:0:00:08.854277
step:  3700, train loss:   0.2, train accuary: 94.59%, val loss :  0.31, val accuary: 91.16%, cost:0:00:10.716808
step:  3800, train loss:  0.21, train accuary: 94.25%, val loss :  0.31, val accuary: 91.18%, cost:0:00:12.528761
step:  3900, train loss:  0.21, train accuary: 94.80%, val loss :  0.31, val accuary: 91.40%, cost:0:00:14.358694
eval test data
loss:  0.27, accuary: 92.09%, cost:0:00:15.470319
```



#### 2 TextRnn

```shell
python3.5 -m "text_classification.main" -model textrnn -save_path save/textrnn -epoch_num 5
```

```shell
create model

 --------------------
TextRNN : parms
TextRnn/embedding:0 (5000, 128)
TextRnn/Rnn/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel:0 (192, 128)
TextRnn/Rnn/rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias:0 (128,)
TextRnn/Rnn/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel:0 (192, 64)
TextRnn/Rnn/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias:0 (64,)
TextRnn/dense/kernel:0 (64, 10)
TextRnn/dense/bias:0 (10,)
-------------------- 

load data set
Epoch: 1
step:   100, train loss:   2.2, train accuary: 18.56%, val loss :   2.1, val accuary: 20.84%, cost:0:00:34.051950
step:   200, train loss:   2.1, train accuary: 26.08%, val loss :   2.1, val accuary: 21.56%, cost:0:01:07.838350
step:   300, train loss:   1.9, train accuary: 31.70%, val loss :   1.8, val accuary: 31.06%, cost:0:01:41.759550
step:   400, train loss:   1.8, train accuary: 34.08%, val loss :   1.9, val accuary: 28.04%, cost:0:02:15.653036
step:   500, train loss:   1.6, train accuary: 41.55%, val loss :   1.5, val accuary: 46.80%, cost:0:02:49.297939
step:   600, train loss:   1.3, train accuary: 52.55%, val loss :   1.6, val accuary: 48.78%, cost:0:03:23.266585
step:   700, train loss:   1.2, train accuary: 58.94%, val loss :   1.1, val accuary: 60.48%, cost:0:03:57.252852
Epoch: 2
step:   800, train loss:  0.88, train accuary: 69.36%, val loss :  0.97, val accuary: 64.06%, cost:0:00:11.253570
step:   900, train loss:  0.75, train accuary: 76.33%, val loss :  0.89, val accuary: 69.96%, cost:0:00:45.180775
step:  1000, train loss:  0.62, train accuary: 82.56%, val loss :  0.82, val accuary: 77.70%, cost:0:01:19.110621
step:  1100, train loss:  0.54, train accuary: 85.58%, val loss :  0.74, val accuary: 78.06%, cost:0:01:53.019108
step:  1200, train loss:  0.48, train accuary: 88.23%, val loss :  0.49, val accuary: 86.38%, cost:0:02:27.000502
step:  1300, train loss:  0.43, train accuary: 89.41%, val loss :  0.45, val accuary: 88.58%, cost:0:03:01.183978
step:  1400, train loss:  0.37, train accuary: 91.00%, val loss :  0.42, val accuary: 88.12%, cost:0:03:34.768332
step:  1500, train loss:  0.38, train accuary: 90.30%, val loss :  0.41, val accuary: 88.46%, cost:0:04:08.398122
Epoch: 3
step:  1600, train loss:  0.31, train accuary: 93.23%, val loss :  0.43, val accuary: 88.48%, cost:0:00:16.136509
step:  1700, train loss:  0.29, train accuary: 92.81%, val loss :  0.39, val accuary: 89.34%, cost:0:00:49.646578
step:  1800, train loss:   0.3, train accuary: 92.36%, val loss :  0.41, val accuary: 88.92%, cost:0:01:23.859286
step:  1900, train loss:  0.27, train accuary: 93.34%, val loss :  0.35, val accuary: 89.88%, cost:0:01:57.909257
step:  2000, train loss:  0.26, train accuary: 93.48%, val loss :  0.37, val accuary: 89.22%, cost:0:02:31.951416
step:  2100, train loss:   0.3, train accuary: 92.09%, val loss :  0.42, val accuary: 87.84%, cost:0:03:05.799133
step:  2200, train loss:  0.27, train accuary: 93.34%, val loss :  0.38, val accuary: 89.26%, cost:0:03:39.777590
step:  2300, train loss:  0.24, train accuary: 94.09%, val loss :  0.35, val accuary: 90.10%, cost:0:04:13.197220
Epoch: 4
step:  2400, train loss:   0.2, train accuary: 95.52%, val loss :   0.3, val accuary: 91.94%, cost:0:00:21.144021
step:  2500, train loss:  0.18, train accuary: 95.62%, val loss :  0.32, val accuary: 91.62%, cost:0:00:54.737925
step:  2600, train loss:  0.19, train accuary: 95.33%, val loss :  0.39, val accuary: 89.18%, cost:0:01:28.416508
step:  2700, train loss:  0.19, train accuary: 95.27%, val loss :  0.35, val accuary: 89.66%, cost:0:02:01.925069
step:  2800, train loss:  0.16, train accuary: 95.95%, val loss :  0.28, val accuary: 92.36%, cost:0:02:35.620588
step:  2900, train loss:   0.2, train accuary: 95.06%, val loss :  0.32, val accuary: 91.22%, cost:0:03:09.146150
step:  3000, train loss:  0.19, train accuary: 95.31%, val loss :  0.29, val accuary: 92.38%, cost:0:03:42.562971
step:  3100, train loss:  0.22, train accuary: 94.97%, val loss :  0.29, val accuary: 92.28%, cost:0:04:16.138834
Epoch: 5
step:  3200, train loss:  0.15, train accuary: 96.42%, val loss :  0.29, val accuary: 92.02%, cost:0:00:26.234004
step:  3300, train loss:  0.17, train accuary: 95.89%, val loss :  0.35, val accuary: 90.48%, cost:0:01:00.304539
step:  3400, train loss:  0.15, train accuary: 96.25%, val loss :  0.31, val accuary: 92.16%, cost:0:01:34.078860
step:  3500, train loss:  0.15, train accuary: 96.27%, val loss :  0.28, val accuary: 92.38%, cost:0:02:07.846796
step:  3600, train loss:  0.17, train accuary: 95.75%, val loss :  0.36, val accuary: 90.72%, cost:0:02:41.827863
step:  3700, train loss:  0.15, train accuary: 96.27%, val loss :  0.26, val accuary: 93.08%, cost:0:03:15.554558
step:  3800, train loss:  0.15, train accuary: 96.34%, val loss :  0.25, val accuary: 93.24%, cost:0:03:49.024171
step:  3900, train loss:  0.14, train accuary: 96.41%, val loss :  0.29, val accuary: 92.30%, cost:0:04:22.491117
eval test data
loss:  0.23, accuary: 93.86%, cost:0:04:37.632397
```



#### 3 TextCnn

```shell
python3.5 -m "text_classification.main" -model textcnn -save_path save/textcnn -epoch_num 5
```

```shell
create model

 --------------------
TextCNN : parms
Fasttext/embedding:0 (5000, 128)
Fasttext/CNN/conv2d/kernel:0 (5, 128, 1, 128)
Fasttext/CNN/conv2d/bias:0 (128,)
Fasttext/dense/kernel:0 (128, 10)
Fasttext/dense/bias:0 (10,)
-------------------- 

load data set
Epoch: 1
step:   100, train loss:   1.8, train accuary: 44.06%, val loss :   1.2, val accuary: 72.78%, cost:0:00:04.801294
step:   200, train loss:  0.68, train accuary: 81.02%, val loss :  0.69, val accuary: 80.74%, cost:0:00:08.525310
step:   300, train loss:  0.48, train accuary: 85.89%, val loss :   0.5, val accuary: 84.88%, cost:0:00:12.036122
step:   400, train loss:  0.36, train accuary: 89.66%, val loss :  0.42, val accuary: 87.58%, cost:0:00:15.809756
step:   500, train loss:  0.31, train accuary: 91.27%, val loss :  0.33, val accuary: 90.22%, cost:0:00:19.548510
step:   600, train loss:  0.26, train accuary: 92.39%, val loss :  0.29, val accuary: 91.58%, cost:0:00:23.657402
step:   700, train loss:  0.27, train accuary: 92.31%, val loss :  0.26, val accuary: 92.30%, cost:0:00:27.475161
Epoch: 2
step:   800, train loss:  0.18, train accuary: 94.97%, val loss :  0.27, val accuary: 92.46%, cost:0:00:01.377078
step:   900, train loss:  0.18, train accuary: 94.73%, val loss :  0.25, val accuary: 92.76%, cost:0:00:05.129438
step:  1000, train loss:  0.17, train accuary: 94.75%, val loss :  0.25, val accuary: 92.70%, cost:0:00:08.793668
step:  1100, train loss:  0.17, train accuary: 95.19%, val loss :  0.25, val accuary: 92.12%, cost:0:00:12.762015
step:  1200, train loss:  0.16, train accuary: 94.98%, val loss :  0.22, val accuary: 93.70%, cost:0:00:16.546919
step:  1300, train loss:  0.15, train accuary: 95.69%, val loss :  0.21, val accuary: 93.98%, cost:0:00:20.877066
step:  1400, train loss:  0.16, train accuary: 95.31%, val loss :   0.2, val accuary: 93.88%, cost:0:00:25.291857
step:  1500, train loss:  0.15, train accuary: 95.41%, val loss :  0.24, val accuary: 92.82%, cost:0:00:29.547380
Epoch: 3
step:  1600, train loss:   0.1, train accuary: 97.09%, val loss :  0.21, val accuary: 93.76%, cost:0:00:01.791284
step:  1700, train loss:  0.11, train accuary: 96.73%, val loss :  0.23, val accuary: 92.98%, cost:0:00:05.452701
step:  1800, train loss:  0.11, train accuary: 96.42%, val loss :  0.22, val accuary: 93.36%, cost:0:00:09.460258
step:  1900, train loss:  0.11, train accuary: 96.69%, val loss :  0.25, val accuary: 92.36%, cost:0:00:13.329682
step:  2000, train loss:   0.1, train accuary: 96.77%, val loss :  0.21, val accuary: 93.84%, cost:0:00:16.989272
step:  2100, train loss:  0.11, train accuary: 96.55%, val loss :  0.22, val accuary: 93.34%, cost:0:00:20.480673
step:  2200, train loss:  0.12, train accuary: 96.36%, val loss :  0.23, val accuary: 92.76%, cost:0:00:24.517457
step:  2300, train loss:  0.12, train accuary: 96.52%, val loss :  0.21, val accuary: 93.44%, cost:0:00:28.338475
Epoch: 4
step:  2400, train loss: 0.076, train accuary: 97.86%, val loss :  0.18, val accuary: 94.52%, cost:0:00:02.856666
step:  2500, train loss: 0.077, train accuary: 97.58%, val loss :  0.19, val accuary: 94.72%, cost:0:00:07.318107
step:  2600, train loss: 0.083, train accuary: 97.50%, val loss :   0.2, val accuary: 94.26%, cost:0:00:11.614801
step:  2700, train loss: 0.087, train accuary: 97.28%, val loss :   0.2, val accuary: 94.22%, cost:0:00:16.110494
step:  2800, train loss: 0.077, train accuary: 97.70%, val loss :   0.2, val accuary: 94.48%, cost:0:00:20.575370
step:  2900, train loss: 0.078, train accuary: 97.53%, val loss :  0.26, val accuary: 93.08%, cost:0:00:25.102530
step:  3000, train loss: 0.083, train accuary: 97.31%, val loss :  0.24, val accuary: 93.82%, cost:0:00:29.608956
step:  3100, train loss: 0.089, train accuary: 97.45%, val loss :  0.22, val accuary: 94.00%, cost:0:00:34.041156
Epoch: 5
step:  3200, train loss: 0.058, train accuary: 98.24%, val loss :  0.22, val accuary: 93.78%, cost:0:00:03.429175
step:  3300, train loss: 0.057, train accuary: 98.20%, val loss :   0.2, val accuary: 94.34%, cost:0:00:08.040378
step:  3400, train loss: 0.062, train accuary: 98.12%, val loss :  0.22, val accuary: 93.86%, cost:0:00:12.500552
step:  3500, train loss: 0.058, train accuary: 98.14%, val loss :   0.2, val accuary: 94.74%, cost:0:00:17.083240
step:  3600, train loss: 0.062, train accuary: 97.98%, val loss :  0.21, val accuary: 94.30%, cost:0:00:21.373843
step:  3700, train loss: 0.072, train accuary: 97.84%, val loss :   0.2, val accuary: 94.46%, cost:0:00:25.535575
step:  3800, train loss:  0.07, train accuary: 97.77%, val loss :  0.21, val accuary: 93.92%, cost:0:00:29.945531
step:  3900, train loss: 0.058, train accuary: 98.23%, val loss :  0.21, val accuary: 93.82%, cost:0:00:34.236629
eval test data
loss:  0.18, accuary: 95.07%, cost:0:00:36.260877
```

#### 4 三个模型的训练结果图如下:

![train_result](https://github.com/jieguangzhou/TextClassification/blob/master/images/train_result.png)