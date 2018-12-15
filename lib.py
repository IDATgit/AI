import numpy as np


class Neuron:
    def __init__(self, input_size, random_weights=True):
        self.activation_method = 'relu'
        self.input_size = input_size
        if random_weights:
            self.weights = self.get_rand_weights()
            self.bias = np.random.rand(1)[0]

    def work(self, input):
        return self.activation(np.sum(self.weights*input) + self.bias)

    def activation(self, input):
        if self.activation_method == 'relu':
            return max(0, input)

    def get_rand_weights(self):
        return np.random.rand(self.input_size)

    def update_weights(self, grad, const):
        self.weights = self.weights - grad*self.weights*const


class Layer:
    def __init__(self, input_size, layer_size):
        self.layer_size = layer_size
        self.input_size = input_size
        self.neurons = [Neuron(input_size) for i in range(layer_size)]

    def work(self, input):
        output_vec = np.zeros(self.layer_size)
        for idx, neuron in enumerate(self.neurons):
            output_vec[idx] = neuron.work(input)


class Net:
    def __init__(self,input_size, layers_count, layer_size, cost='quad'):
        first_layer = Layer(input_size, layer_size[0])
        self.layers = [first_layer] + [Layer(layer_size[i-1],layer_size[i]) for i in range(1, layers_count)]

    def work(self, input):
        current_input = input
        for layer in self.layers:
            output = layer.work(current_input)
            current_input = output
        return output