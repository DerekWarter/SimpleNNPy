import numpy as np
import math

class Perceptron:
    inputs = np.array([0,0,1], dtype=np.float64)
    weights = np.array([np.random.random(),
                        np.random.random(),1], dtype=np.float64)

    learningRate = .01
    output = 0
    curError = 0 # Used for tracking the effectiveness of training in pyplot.

    def feedForward(self):
        weightedSum = np.dot(self.inputs, self.weights)
        #self.output = math.tanh(weightedSum)
        if (weightedSum > 0):
            self.output = 1
        else:
            self.output = -1

    def learn(self, target):
        self.curError = (target - self.output) # Unweighted loss function.
        index = 0
        for w, i in zip(self.weights, self.inputs):
            neww = w + (self.learningRate * self.curError * i) # Perceptron learning rule.
            self.weights[index] = neww
            index += 1
