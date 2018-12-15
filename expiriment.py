import lib
import numpy as np
import matplotlib.pyplot as plt


def neuron_test():
    input_size = 1
    # we want a single neuron to calculate 2*input
    rand_vec = np.random.rand(1000)
    for step in [0.01, 0.02, 0.03]:
        neuron = lib.Neuron(input_size)
        cost_lst = []
        for input in rand_vec:
            output = neuron.work(input)
            cost = (2*input+5-output)**2
            d_cost_d_out = -2*(2*input+5-output)
            grad_w = neuron.weight_derivative()*d_cost_d_out
            grad_b = neuron.bias_derivative()*d_cost_d_out
            neuron.update_weights([grad_w], step)
            neuron.update_bias(grad_b, step)
            cost_lst.append(cost)
        plt.plot(cost_lst, label=str(step))
    plt.legend()
    plt.title('cost')
    plt.show()


if __name__ == '__main__':
    neuron_test()