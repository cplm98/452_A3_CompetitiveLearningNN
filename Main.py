from Neural_Network import Neural_Network
from DataManager import DataManager
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

#kohonen weights final and init are the same so i'm not copying them right I kohonen_guess
#k-means wieghts are all the same value which makes me think they aren't being updated properly

DataManager = DataManager()
test_k_means = Neural_Network([3,2])
test_kohonen = Neural_Network([3,2])

#***** K_Means Results*****#
test_k_means.initialize_weights(DataManager)
k_means_init_weights = copy.deepcopy(test_k_means.layer_weights)
test_k_means.k_means_learn(DataManager.data)
k_means_final_weights = copy.deepcopy(test_k_means.layer_weights)
clusters = test_k_means.k_means_clusters
cluster0 = clusters["cluster0"]
cluster0_data = (cluster0[:,0], cluster0[:,1], cluster0[:,2])
cluster1 = clusters["cluster1"]
cluster1_data = (cluster1[:,0], cluster1[:,1], cluster1[:,2])
data = (cluster0_data, cluster1_data)
print("\nK-Means Initial Weights: ", k_means_init_weights)
print("\nK-Means Final Weights: ", k_means_final_weights)

color = ("r", "b")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print("\nK_means epoch: ", test_k_means.k_means_epoch)

for set_, color in zip(data, color):
    x, y, z = set_
    ax.scatter(x, y, z, c=color)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title("K-means Squared")


#*****Kohonen Results*****#
test_kohonen.initialize_weights(DataManager)
kohonen_init_weights = copy.deepcopy(test_kohonen.layer_weights)
print("\nStarting Kohonen Learn, this will take a second")
for i in range(500): # number of epochs
    for input_ in DataManager.data: #loop through each data point
        test_kohonen.kohonen_learn(input_) #adjust weights on each point

kohonen_final_weights = copy.deepcopy(test_kohonen.layer_weights)
print("Kohonen initial weights: ", test_kohonen.layer_weights)
print("Kohonen Final Weights: ", kohonen_final_weights)
x_0 = []
y_0 = []
z_0 = []
x_1 = []
y_1 = []
z_1 = []
for input_ in DataManager.data:
    guess = test_kohonen.kohonen_guess(input_)
    if guess == 0:
        x_0.append(input_[0])
        y_0.append(input_[1])
        z_0.append(input_[2])
    else:
        x_1.append(input_[0])
        y_1.append(input_[1])
        z_1.append(input_[2])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x_0, y_0, z_0, c="r")
ax1.scatter(x_1, y_1, z_1, c="b")

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_title("Kohonen Results")

plt.show()
