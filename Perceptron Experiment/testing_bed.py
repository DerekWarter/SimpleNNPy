import matplotlib.pyplot as plt
import numpy as np
import math

from perceptron_obj import Perceptron

## TESTING CODE

def funcv(x):
    return 2 * x

axelrod = Perceptron()
axelrod.learningRate = .0001
# coordinates of inputs above testing line.
abovey = []
abovex = []
# coordinates of inputs below testing line.
belowy = []
belowx = []
# Average error at current iteration.
erravg = 0
erravglist = []
# Average error after training complete.
truavg = 0

i = 0
trynum = 12000
while i < trynum:
    axelrod.inputs = [np.random.uniform(-100,100),
                      np.random.uniform(-100,100),1] #Last input is bias.

    axelrod.feedForward()
    if axelrod.inputs[1] > funcv(axelrod.inputs[0]):
        axelrod.learn(1)
        abovey.append(axelrod.inputs[1])
        abovex.append(axelrod.inputs[0])

    else:
        axelrod.learn(-1)
        belowy.append(axelrod.inputs[1])
        belowx.append(axelrod.inputs[0])

    prettyErr = 0
    if axelrod.curError != 0:
        prettyErr = 100
    truavg += prettyErr
    erravg = truavg/(i+1)
    erravglist.append(erravg)
    print("AVERAGE ERROR THUS FAR: ", erravg, "%")
    i += 1

truavg /= trynum
print("FINAL ERROR PERCENTAGE: ", truavg, "%")
plt.rcParams['toolbar'] = 'None'
plt.subplot(2,1,1)
plt.plot(erravglist, label="Average at Iteration")
plt.plot([0,trynum],[truavg,truavg], label="Final Average")
plt.ylabel('Error %')
plt.legend()

plt.subplot(2,1,2)
plt.ylabel('Training Points')
plt.plot([-100,100],[-200,200])
plt.plot([abovex],[abovey], 'ro', markersize=1)
plt.plot([belowx],[belowy],'bo', markersize=1)
plt.show()
