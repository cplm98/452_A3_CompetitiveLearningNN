from Neural_Network import Neural_Network
from DataManager import DataManager
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

DataManager = DataManager() # create DataManager
test_k_means = Neural_Network([3,2]) # create NN for k-Means
test_kohonen = Neural_Network([3,2]) # create NN for kohonen

#***** K_Means Results*****#
test_k_means.initialize_weights(DataManager) # initialize weights
k_means_init_weights = copy.deepcopy(test_k_means.layer_weights) # save initial weights
test_k_means.k_means_learn(DataManager.data) #
k_means_final_weights = copy.deepcopy(test_k_means.layer_weights)
print("\nK-Means Initial Weights: ", k_means_init_weights)
print("\nK-Means Final Weights: ", k_means_final_weights)
print("\nK_means epoch: ", test_k_means.k_means_epoch)

# coordinate lists
x_0 = [] # cluster 0
y_0 = []
z_0 = []
x_1 = [] # cluster 1
y_1 = []
z_1 = []
for input_ in DataManager.data:
    guess = test_k_means.guess(input_)
    if guess == 0:
        x_0.append(input_[0])
        y_0.append(input_[1])
        z_0.append(input_[2])
    else:
        x_1.append(input_[0])
        y_1.append(input_[1])
        z_1.append(input_[2])

color = ("r", "b")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_0, y_0, z_0, c="r")
ax.scatter(x_1, y_1, z_1, c="b")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title("K-means Squared")


#*****Kohonen Results*****#
test_kohonen.initialize_weights(DataManager)
kohonen_init_weights = copy.deepcopy(test_kohonen.layer_weights)
print("\nStarting Kohonen Learn, this will take a second...\n")
prev_error = 100000 # keeps track of previous epoch error
err_sum = 0 # keeps track of per epoch error
j = 0 # Kohonen epoch counter
for i in range(500): # number of epochs
    j = i
    if np.square(prev_error - err_sum) < .000001: # if level of accuracy is achieved break
        break
    else:
        prev_error = err_sum # update previous error
        err_sum = 0 # reset current error
        for input_ in DataManager.data: #loop through each data point
            err = test_kohonen.kohonen_learn(input_) #adjust weights on each point
            err_sum += err.sum() # sum error

print("Kohonen Error", err_sum)
print("Kohonen Epoch Count: ", j)

kohonen_final_weights = copy.deepcopy(test_kohonen.layer_weights)
print("Kohonen initial weights: ", kohonen_init_weights)
print("Kohonen Final Weights: ", kohonen_final_weights)
# coordinate lists
x_2 = [] # cluster 2
y_2 = []
z_2 = []
x_3 = [] # cluster 3
y_3 = []
z_3 = []
for input_ in DataManager.data:
    guess = test_kohonen.guess(input_)
    if guess == 0:
        x_2.append(input_[0])
        y_2.append(input_[1])
        z_2.append(input_[2])
    else:
        x_3.append(input_[0])
        y_3.append(input_[1])
        z_3.append(input_[2])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x_2, y_2, z_2, c="r")
ax1.scatter(x_3, y_3, z_3, c="b")

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_title("Kohonen Results")

plt.show()
