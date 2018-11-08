import numpy as np

class FeedForwardNeuralNetwork:
    def __init__(self, learning_rate, hidden_layer_sizes, tol=0.0001, batch_size=1, momentum=0):
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.tol = tol
        self.batch_size = batch_size
        self.momentum = momentum
        self._check_validity()
        self.coefs_ = []
        self.intercepts_ = []

    def _check_validity(self):
        if (not self._is_hidden_layer_sizes_valid()):
            raise(AttributeError('Hidden layer sizes %s is invalid' % (self.hidden_layer_sizes,)))
    
    def _is_hidden_layer_sizes_valid(self):
        _valid_length = (len(self.hidden_layer_sizes) in range (0, 11))
        _valid_content = all(x > 0 for x in self.hidden_layer_sizes)
        return _valid_length and _valid_content

    def _initialize_coefs(self, data):
        input_neuron_size = len(data[0])
        output_neuron_size = 1
        neuron_sizes = [input_neuron_size] + self.hidden_layer_sizes + [output_neuron_size]
        for neuron_index in range (len(neuron_sizes)-1):
            layer_weight = []
            for _ in range(neuron_sizes[neuron_index]):
                neuron_weight = []
                for _ in range(neuron_sizes[neuron_index+1]):
                    neuron_weight.append(0)
                layer_weight.append(neuron_weight)
            layer_weight = np.array(layer_weight)
            self.coefs_.append(layer_weight)

    def _initialize_intercepts(self):
        print('initializing intercepts...')
    
    def _feed_forward_phase(self, data):
        return None

    def _backpropagation(self):
        return None

    def _update_weight(self):
        return None

    def fit(self, data):
        self._initialize_coefs(data)
        print('fitting...')