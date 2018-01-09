# Speech Challenge

TensorFlow Speech Recognition Challenge
[Kaggle Competition Page](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)

## Models
* example_conv
* cnn_trad_fpool3 [http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf]
* esc_cnn [http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf]
* deep_cnn
* deeep_cnn
* deeeep_cnn

## Usage
```
python3 train.py \
--data_url= \
--model_architecture=deeeep_cnn \
--how_many_training_steps=30000,30000,180000 \
--learning_rate=0.001,0.0001,0.001 \
--batch_size=500 \
--start_checkpoint=speech_commands_train/deeeep_cnn.ckpt-200000
```
Training Dataset [https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/train.7z]
```
python3 freeze.py \
--model_architecture=deeeep_cnn \
--start_checkpoint=speech_commands_train/deeeep_cnn.ckpt-240000 \
--output_file=graph.pb
```
```
python3 label_wav.py \
--graph=graph.pb \
--labels=speech_commands_train/deeeep_cnn_labels.txt \
--wav_dir=test_dataset
```
Testing Dataset [https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/test.7z]

## Result
* example_conv (77%)
* cnn_trad_fpool3 (77%)
* esc_cnn (77%)
* deep_cnn (79%)
* deeep_cnn (81%)
* deeeep_cnn (79%)
