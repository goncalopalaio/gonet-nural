# gonet-nural
Experiment in training a simple Tensorflow model using Python, exporting it and reimplementing it in C with minimal dependencies.

Included is an Android project that uses the C code through the NDK to recognize digits drawn on the screen.

![android-sample-gif](https://github.com/goncalopalaio/gonet-nural/blob/master/imgs/android-sample-demo.gif?raw=true)

The network is trained to recognise digits 0-9 using the MNIST dataset.
It only has one convolution layer and the hyperparameters are not really well tuned. Don't expect to achieve good results with this.
The main purpose of the network was to test the C implementation of the network operations (convolution, maxpool, relu, flatten etc).

When training is finished, the network weights are evaluated and written to a file.
The file format is the following:

- single unsigned int - number of dimensions the data has
- sequence of unsigned int - actual dimension values
- sequence of float - weights in row major order.

The input data is processed by reading the weights from file and applying the weigths and operations of the original network to the input.

The Android project uses a custom view where digits can be drawn on the screen. When the user taps analyse, the contents of the view are downsampled and passed through JNI to the C library that will load the weights (from the SDCARD, must be copied there first) and return the digit that was drawn in the view.

The code is not production grade and I make no guarantees that this source code is fit for any purpose.
