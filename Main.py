 # limit is imposed on the strength of each neuron

 # mechanism that allows neurons to compete for the right to
 # respond to a given subset of inputs, such that only one neuron
 # is on at a time, winner-take all neuron

 # basically feature detectors

 # probably start by building a class that creates a multilayer network, that should stay persistent
from Neural_Network import Neural_Network

arr = [2,6]

test = Neural_Network(arr)
print("nodes: ", test.layer_nodes)
for key in test.layer_weights.keys():
    print("weights: ", test.layer_weights[key])
