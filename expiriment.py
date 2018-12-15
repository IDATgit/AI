import lib
import numpy as np
import matplotlib.pyplot as plt


def neuron_test():
    input_size = 1
    # we want the neuron to calculate 2*input
    rand_vec = np.random.rand(100)
    for step in [0.01, 0.1, 0.2, 0.3, 1]:
        neuron = lib.Neuron(input_size)
        cost_lst = []
        for input in rand_vec:
            output = neuron.work(input)
            cost = (2*input-output)**2
            cost_lst.append(cost)
            grad = 2*input*(neuron.weights[0]*input - 2*input)
            neuron.update_weights([grad], 1)
        plt.plot(cost_lst, label=str(step))
    plt.legend()
    plt.title('cost')
    plt.show()


if __name__ == '__main__':
    neuron_test()