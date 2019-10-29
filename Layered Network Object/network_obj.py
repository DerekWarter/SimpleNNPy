import numpy as np
import math
import netwide_funcs as nf

# NODE OBJECT:
# Represents a node within an ANN
# Contains activation function, derivative of activation function,
# an array of inputs from the previous layer output,
# an array of weights associated with inputs,
# the sum of inputs/weights, and an output value.

class Node:
    def __init__(self):
        self.activation_func = nf.relu
        self.activation_deriv = nf.relu_deriv
        self.node_inputs = np.array([1], dtype=np.float64)
        self.node_weights = np.array([1], dtype=np.float64)
        self.node_grads = np.array([0], dtype=np.float64)
        self.node_summed_input = 1
        self.node_output = 1

# NEURAL NETWORK OBJECT:
# Represents an ANN using stochastic gradient descent.
# Contains network inputs, number of layers (incl. output), an output vector,
# alpha, and a netwide bias value.

class NN:
    def __init__(self):
        self.net_inputs = np.array([0,0,1], dtype=np.float64)
        self.net_num_layer = 2
        self.net_layer = [[Node() for n in range(1)]
                                  for l in range(self.net_num_layer)]
        self.net_output = np.array([n.node_output for n in self.net_layer[self.net_num_layer - 1]],
                                   dtype=np.float64)
        self.err_func = nf.squared_error
        self.err_deriv = nf.squared_error_deriv
        self.learning_rate = 0.1
        self.net_bias = 1

    # RESTRUCTURE: Reinitializes the network after nodes/layers have been added or removed.
    # Ensures node input/weight vectors have correct shape.

    def restructure(self):
        first_layer = True
        prev_layer = self.net_layer[0]
        for l in self.net_layer:
            for n in l:
                if first_layer:
                    n.node_inputs = np.array(self.net_inputs, dtype=np.float64)
                    n.node_weights = np.array([np.random.random() for w in self.net_inputs], dtype=np.float64)
                    n.node_grads = np.array([0 for w in self.net_inputs], dtype=np.float64)
                else:
                    n.node_inputs = np.array([pln.node_output for pln in prev_layer], dtype=np.float64)
                    n.node_weights = np.array([np.random.random() for w in prev_layer], dtype=np.float64)
                    n.node_grads = np.array([0 for w in prev_layer], dtype=np.float64)
            prev_layer = l
            first_layer = False
        self.net_num_layer = len(self.net_layer)
        self.net_output = np.array([n.node_output for n in self.net_layer[self.net_num_layer - 1]], dtype=np.float64)

    # ADD_BIAS: Adds a bias across the network.

    def add_bias(self):
        for l in self.net_layer:
            for n in l:
                n.node_inputs = np.append(n.node_inputs, [self.net_bias])
                n.node_weights = np.append(n.node_weights, [self.net_bias])

    # FEED_FORWARD: Runs a feed forward pass on the neural network.

    def feed_forward(self):
        prev_layer = self.net_layer[0]
        first_layer = True

        for l in self.net_layer:
            for n in l:
                if first_layer:
                    n.node_inputs = np.array(self.net_inputs, dtype=np.float64)
                    n.node_inputs = np.append(n.node_inputs, [self.net_bias])
                if not first_layer:
		    # Connect this layer to outputs of previous layer. Restore bias.
                    n.node_inputs = np.array([pln.node_output for pln in prev_layer], dtype=np.float64)
                    n.node_inputs = np.append(n.node_inputs, [self.net_bias])
                n.node_summed_input = np.dot(n.node_inputs, n.node_weights)
            # Because some act. functions rely on outputs of other nodes, we must get all dot products before continuing. 2 passes required.
            # Activation functions take the output for the whole layer, so we only need to run the activation of the first node in each layer,
            # passing a list of all weighted inputs.
	        # -> Consider making activation functions an NN class property instead.
            layer_summed_input = [n.node_summed_input for n in l]
            layer_output = l[0].activation_func(layer_summed_input)
            out_count = 0
            for n in l:
                n.node_output = layer_output[out_count]
                out_count += 1
            prev_layer = l
            first_layer = False

        self.net_output = np.array([n.node_output for n in self.net_layer[self.net_num_layer - 1]],
                                   dtype=np.float64)

    # OUTPUT LAYER ERROR:

    def net_error_signal(self, T):
        ol = self.net_layer[self.net_num_layer - 1]
        layer_summed_input = [n.node_summed_input for n in ol]
        return self.err_deriv(self.net_output, T) * ol[0].activation_deriv(layer_summed_input)

    # HIDDEN LAYER ERROR:

    def hidden_error_signal(self, this_layer, next_layer_error, leading_weights):
        layer_output = [n.node_output for n in this_layer]
        layer_err = []
        for n_this in range(len(this_layer)):
            this_error = 0
            for n_next in range(len(next_layer_error)):
                this_error += next_layer_error[n_next] * leading_weights[n_next][n_this]
            layer_err.append(this_error)
        layer_err = np.array(layer_err, dtype=np.float64)
        layer_err = layer_err * this_layer[0].activation_deriv(layer_output)
        return layer_err

    # LAYER GRAD: Compute node gradient for a given layer.

    def layer_grad(self, this_layer, this_error):
        layer_inputs = [n.node_inputs for n in this_layer]
        layer_grad = []
        for n in range(len(this_layer)):
            layer_grad.append([])
            for i in range(len(layer_inputs[n])):
                layer_grad[n].append(this_error[n] * layer_inputs[n][i])
        return layer_grad

    # NETWORK GRAD: Compute node gradients for entire network.

    def network_grad(self, T):
        last_layer = True
        net_grad = []
        for l in reversed(range(self.net_num_layer)):
            this_layer = self.net_layer[l]
            if last_layer:
                this_error = self.net_error_signal(T)
                last_layer = False
            else:
                this_error = self.hidden_error_signal(this_layer, next_error, leading_weights)
            this_grad = self.layer_grad(this_layer, this_error)
            net_grad.append(this_grad)
            next_error = this_error
            leading_weights = [n.node_weights for n in this_layer]
        net_grad = np.flip(net_grad, axis=0)
        return net_grad

    # DELTA WEIGHTS: Get new weights/update weights for network.

    def delta_weights(self, T, update=False):
        network_grad = self.network_grad(T)

        for l in range(self.net_num_layer):
            this_layer = self.net_layer[l]
            for n in range(len(this_layer)):
                this_node = this_layer[n]
                for w in range(len(this_node.node_weights)):
                    w_delta = self.learning_rate * network_grad[l][n][w]
                    if update:
                        this_node.node_weights[w] -= w_delta

    # PRINT NETWORK: Print data of all network nodes.

    def print_network(self):
        l_i = 0
        for l in self.net_layer:
            n_i = 0
            for n in l:
                print("LAYER: ", l_i, "\n NODE: ", n_i, "\n INPUTS: ",
                      n.node_inputs, "\n WEIGHTS: ", n.node_weights,
                      "\n OUTPUT: ", n.node_output, "\n SUMMED INPUT: ", n.node_summed_input, "\n\n")
                n_i += 1
            l_i += 1
