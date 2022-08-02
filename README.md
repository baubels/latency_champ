# Fast stock price prediction on microcontrollers

We seek to predict single ticker prices on a microcontroller. Unfortunately, in the competition, the small device used for running code only had sklearn and numpy installed, so for that reason a simple MLP neural network using sklearn had to be made. It produced a working implementation, which somehow got the highest prediction rate in the competition. On the other hand, of the top contenders, it was the slowest! On average, the inference time on a bunch of examples was 0.9 seconds vs. the 0.2 seconds of the others. 

Rerunning the results (in house) by using the optimisations I had planned to use (specifically that of .tflite model generation), inference time decreased to 0.045 seconds (from 0.9 seconds), therefore giving both the quickest and most accurate stock price predictor of the competition.

With the tflite model was already built, a 20x speed improvement was found, with a further model reduction size of 100x.

The neural net topology is as follows:

![model tflite](https://user-images.githubusercontent.com/42357923/120664774-1f4f9080-c483-11eb-9f3b-398c0f793915.png)
