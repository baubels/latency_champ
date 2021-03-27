import os
import numpy as np
# sudo apt-get install python3-tflite-runtime
import tflite_runtime.interpreter as tflite
from sys import stdin


# load interpreter from file
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# classify terminal input
for line in stdin:
    if line == '': 
        break
    d=[float(x) for x in line.split(',')]

    # preprocessing
    d=np.array(d[-101:])
    d[1:] -= d[:-1]
    d = np.delete(d, [0])
    d[d < 0] = 0
    d[d > 0] = 1
    d = d.astype(np.float32)
    x = np.reshape(d, (1, 100, 1))
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    print(int(interpreter.get_tensor(output_details[0]['index'])))
