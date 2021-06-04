# latency_champ

We seek to predict single ticker prices on a microcontroller. Unfortunately, in the competition, the small device used for running code only had sklearn and numpy installed, so for that reason a simple MLP neural network using sklearn had to be made. It produced a working implementation, which somehow got the highest prediction rate in the competition. On the other hand, of the top contenders, it was the slowest! On average, the inference time on a bunch of examples was 0.9 seconds vs. the 0.2 seconds of the others.

However, as the tflite model was already built, comparing its performance to other models was an easy next step. On my personal computer, a 20x speed improvement was found, further by a reduction in size of model by 100x.

The topology of my tflite model can be found below:

![model tflite](https://user-images.githubusercontent.com/42357923/120664774-1f4f9080-c483-11eb-9f3b-398c0f793915.png)
