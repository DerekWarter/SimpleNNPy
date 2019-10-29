import matplotlib.pyplot as plt
import import_lists as il
import build_net as bn

print("TRAINING: \n")

net = bn.lc_net()
wlist = il.import_lists("eng_out.txt", "ger_out.txt")

correct = 0
err = 0
err_last_x = 0
erravg = 0
i = 0
#for i in range(num_steps):
while(True):
    tpair = il.training_pair(wlist)
    input = tpair[0]
    targ = [tpair[1]]
    net.net_inputs = input
    net.feed_forward()
    if abs(net.net_output[0] - targ[0]) < 1:
        correct += 1
    else:
        err += 1
        err_last_x += 1
    erravg = err/(i + 1)
    if (i % 1000 == 0) and (i > 100):
        print("NETWORK ACCURACY: %.2f" % ((1 - erravg) * 100), "%")
        last_thousand = (1 - (err_last_x/1000)) * 100
        print("NETWORK ACCURACY LAST 1000 ITERS: %.2f" % last_thousand, "%")
        print ("C: ", correct, " E: ", err_last_x)
        if last_thousand > 84:
            break
        else:
            err_last_x = 0
            correct = 0

    net.delta_weights(targ, True)
    i += 1

net.net_inputs = il.to_vector("blunder")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("willow")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("whimsical")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("bureaucrat")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("blanchierende")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("festnetzflatrat")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("festpilger")
net.feed_forward()
print(net.net_output)

net.net_inputs = il.to_vector("festplattenmark")
net.feed_forward()
print(net.net_output)
