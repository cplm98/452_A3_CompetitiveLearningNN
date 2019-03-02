import numpy as np
import random


# create class that takes in an array of number of layers and number of nodes per layers

# I'm calculating this distance wrong
def euclid_dist(weight_vec,input_vec):
    dist = np.sqrt(np.sum(np.square(np.subtract(input_vec, weight_vec))))
    return dist

class Neural_Network():
    def __init__(self, arr):
        self.num_layers = len(arr)
        self.c = .0001 # learning rate
        # generate nodes per layer
        self.layer_nodes = {}
        for i in range(self.num_layers):
            self.layer_nodes["layer" + str(i)] = arr[i]

        # generate weights
        self.layer_weights = {}
        for i, key in enumerate(self.layer_nodes.keys()):
            if key == "layer0":
                prev_key = key
            else:
                self.layer_weights["weights" + str(i)] = np.random.uniform(
                    low=-.5, high=.5, size=(self.layer_nodes[key], self.layer_nodes[prev_key]))
                prev_key = key

    def kohonen_guess(self, input_arr):
        node_dists = []
        for key in self.layer_weights.keys():
            weights_mat = self.layer_weights[key]
            num_rows = weights_mat.shape[0]
            for row in range(num_rows):
                weight_row = weights_mat[row,]
                dist = euclid_dist(weight_row, input_arr)
                node_dists.append(dist)
        winner = np.argmin(node_dists) # returns the index of the smallest distance
        return winner

    def kohonen_learn(self, input_arr):
        node_dists = []
        for key in self.layer_weights.keys():
            weights_mat = self.layer_weights[key]
            num_rows = weights_mat.shape[0]
            # print("weights key ", key)
            for row in range(num_rows):
                weight_row = weights_mat[row,]
                dist = euclid_dist(weight_row, input_arr)
                node_dists.append(dist)
        print(node_dists)
        winner = np.argmin(node_dists) # index of the smallest distance
        # print("\n winning node: ", winner)
        # print("\n Winner: ", weights_mat[winner,], "\n")
        # finds smallest distance properly

        # adjust winning weight
        delta_w = self.c*(np.subtract(input_arr, weights_mat[winner]))
        # print("delta: ", delta_w)
        weights_mat[winner,] = weights_mat[winner,] +  delta_w # this links back and changes the matrix in self.layer_weights

# print(euclid_dist([5,3,2,1], [3,4,5,6]))
# test = Neural_Network([3,2])
# data = [3,4,5]
# print(test.layer_weights)
# test.kohonen_learn(data)
# print("\n\n post: ", test.layer_weights)
