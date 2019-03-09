import numpy as np
import random
import copy

# create class that takes in an array of number of layers and number of nodes per layers

# I'm calculating this distance wrong


def euclid_dist(weight_vec, input_vec):
    dist = np.sqrt(np.sum(np.square(np.subtract(input_vec, weight_vec))))
    return dist

def error_(point, centroid):
    err = np.square(np.subtract(point, centroid))
    return err

class Neural_Network():
    def __init__(self, arr):
        self.num_layers = len(arr)
        self.c = .0001  # learning rate
        self.k_means_clusters = {}
        self.k_means_epoch = 0
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

    def initialize_weights(self, DataManager):
        # initialize starting weights to random points from data set
        num_rows = DataManager.data.shape[0]
        init_idx = random.sample(range(0,num_rows), 2)
        self.layer_weights["weights1"] = np.array([list(DataManager.data[init_idx[0]]), list(DataManager.data[init_idx[1]])])

    def guess(self, input_arr):
        node_dists = []
        for key in self.layer_weights.keys():
            weights_mat = self.layer_weights[key]
            num_rows = weights_mat.shape[0]
            for row in range(num_rows):
                weight_row = weights_mat[row, ]
                dist = euclid_dist(weight_row, input_arr)
                node_dists.append(dist)
        # returns the index of the smallest distance
        winner = np.argmin(node_dists)
        return winner

    def kohonen_learn(self, input_arr):
        node_dists = []  # list of point distances from centroids
        for key in self.layer_weights.keys():  # for set of weights in layer_weights
            weights_mat = self.layer_weights[key]
            # get number of rows of weight matrix
            num_rows = weights_mat.shape[0]
            for row in range(num_rows):  # for each row, aka each nodes set of weights
                weight_row = weights_mat[row, ]
                # calculate distance between centroid (node position) and input point position
                dist = euclid_dist(weight_row, input_arr)
                node_dists.append(dist)  # add distance to no_dists
        winner = np.argmin(node_dists)  # index of the smallest distance
        err = error_(input_arr, weights_mat[winner, ])
        # adjust winning weight
        delta_w = self.c * (np.subtract(input_arr, weights_mat[winner]))
        # this links back and changes the matrix in self.layer_weights
        weights_mat[winner, ] = weights_mat[winner, ] + delta_w
        return err

    def k_means_learn(self, all_inputs):
        # take the only weight layer
        weights_mat = self.layer_weights["weights1"]
        k = weights_mat.shape[0]  # number of nodes aka number of clusters
        clusters = {}
        prev_cluster = {}
        # initialize cluster lists
        for i in range(k):
            clusters["cluster" + str(i)] = []
            prev_cluster["cluster" + str(i)] = []
        while True:
            self.k_means_epoch += 1 # count number of epochs
            for i in range(k):
                clusters["cluster" + str(i)] = []
            # Categorize each point by cluster
            for point in all_inputs:
                node_dists = [] # store distances from point to each centroid
                point = np.array(point)
                for i in range(k):
                    # calculate distance from centroids
                    node_dists.append(euclid_dist(weights_mat[i, ], point))
                winner = np.argmin(node_dists) # winning index is centroid closest to point
                # if empty, initialize dimensions of ndarray
                if clusters["cluster" + str(winner)] == []:
                    clusters["cluster" + str(winner)].append(point)
                else:
                    # use vstack to maintain shape of ndarray and coordinates of each point
                    clusters["cluster" + str(winner)] = np.vstack(
                        (clusters["cluster" + str(winner)], point))
            for i in range(k):
                weights_mat[i, ] = np.mean(clusters["cluster" + str(i)], axis=0) # adjust centroids to mean position of all points
                                                                                 # in its cluster
            # exit if cluster for each centroid doesn't change, meaning if centroid position didn't change
            if np.array_equal(clusters["cluster0"], prev_cluster["cluster0"]) and np.array_equal(clusters["cluster1"], prev_cluster["cluster1"]):
                self.k_means_clusters = copy.deepcopy(clusters)
                break
            else:
                prev_cluster = copy.deepcopy(clusters)
