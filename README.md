# latency_champ

We seek to predict single ticker prices on a microcontroller. Unfortunately, in the competition, the small device running predictions only had sklearn and numpy installed, so for that reason a simple neural network using sklearn (MLP) had to be made. It produced a working implementation with highest prediction in the competition. Unfortunetaly, it was quite slow and had a 0.9 second inference time vs. 0.2 seconds avg.

The tensorflow lite model was the first to be made, and on my computer, I achieved a ~20x speed improvement. Although the microcontroller in the competition was not available to me physically, I believe a similar change in inference speed would have been possible. Further, as my tflite model was 10kb in size, this would have given a 100x reduction in memory consumption.

The neural network is based on a simple non-convolutional dense network of three layers with nodes in the 10s in each layer. The topology of it is as follows:

![model tflite](https://user-images.githubusercontent.com/42357923/120664774-1f4f9080-c483-11eb-9f3b-398c0f793915.png)
