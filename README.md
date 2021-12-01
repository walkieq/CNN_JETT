# CNN_JETT
Tested using **Vivado HLS 2019.02**. 

- Model using Keras/Tensorflow
```python
model = Sequential()
model.add(layers.Conv1D(8, (4), name='conv1', activation='relu', input_shape=(16, 1)))
model.add(layers.Conv1D(8, (4), name='conv2', activation='relu'))
model.add(layers.Conv1D(8, (4), name='conv3', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu', name='fc1'))
model.add(layers.Dense(16, activation='relu', name='fc7'))
model.add(layers.Dense(5, activation='softmax', name='output'))
```
- How to run without GUI
```batch
cd cnn_jet_tagging
vivado_hls -f build.tcl
```
