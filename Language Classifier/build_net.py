import sys
sys.path.insert(0, '../Layered Network Object')
import network_obj as no

def lc_net():
    net = no.NN()

    for hn in range(19):
        net.net_layer[0].append(no.Node())
        net.net_layer[0][hn].activation_func = no.nf.tanh
        net.net_layer[0][hn].activation_deriv = no.nf.tanh_deriv

    net.net_layer[1][0].activation_func = no.nf.tanh
    net.net_layer[1][0].activation_deriv = no.nf.tanh_deriv

    input_shape = []
    for input in range(15*26):
        input_shape.append(0)

    net.net_inputs = input_shape
    net.net_bias = 0
    net.learning_rate = .25

    net.restructure()
    net.add_bias()

    return net
