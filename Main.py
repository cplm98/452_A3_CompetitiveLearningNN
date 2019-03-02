 # limit is imposed on the strength of each neuron

 # mechanism that allows neurons to compete for the right to
 # respond to a given subset of inputs, such that only one neuron
 # is on at a time, winner-take all neuron

 # basically feature detectors

 # probably start by building a class that creates a multilayer network, that should stay persistent
from Neural_Network import Neural_Network
from DataManager import DataManager
import copy
import matplotlib.pyplot as plt
import random
import numpy as np

DataManager = DataManager()
test = Neural_Network([3,2])
# initialize starting weights to random points
num_rows = DataManager.data.shape[0]
init_idx = random.sample(range(0,num_rows), 2)
print("random point", DataManager.data[init_idx[0]])
test.layer_weights["weights1"] = np.array([list(DataManager.data[init_idx[0]]), list(DataManager.data[init_idx[1]])])  # np.array([DataManager.data[init_idx[0]], [DataManager.data[init_idx[1]]]])
print("new weights: ", test.layer_weights)
init_weights = copy.deepcopy(test.layer_weights)
x_1 = []
y_1 =[]
x_2 = []
y_2 = []

j = 0
for i in range(50):
    for input_ in DataManager.data:
        test.kohonen_learn(input_)
        # create data set for plotting movement of centroids
        x = test.layer_weights['weights1'][0,][0]
        y = test.layer_weights['weights1'][0,][1]
        x_1.append(test.layer_weights['weights1'][0,][0])
        y_1.append(test.layer_weights['weights1'][0,][1])

        x = test.layer_weights['weights1'][1,][0]
        y = test.layer_weights['weights1'][1,][1]
        x_2.append(test.layer_weights['weights1'][1,][0])
        y_2.append(test.layer_weights['weights1'][1,][1])

weight1 = (x_1, y_1)
weight2 = (x_2, y_2)
# print("\nx's: ", weight1_x, "\n")
print("initial weights: ", init_weights)
print("final weights: ", test.layer_weights)
data = (weight1, weight2)
color = ("red", "blue")

x_0 = []
y_0 = []

x_1 = []
y_1 = []

for input_ in DataManager.data:
    guess = test.kohonen_guess(input_)
    if guess == 0:
        x_0.append(input_[0])
        y_0.append(input_[1])
    else:
        x_1.append(input_[0])
        y_1.append(input_[1])

res1 = (x_0, y_0)
res2 = (x_1, y_1)
results = (res1, res2)

for data, color in zip(results, color):
    x, y = data
    plt.scatter(x, y, c=color)
plt.show()
