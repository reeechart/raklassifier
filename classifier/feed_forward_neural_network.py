import math
import numpy as np
from scipy.stats import logistic
from copy import deepcopy

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
        self.errors_ = []

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

    def _initialize_intercepts(self, data):
        output_neuron_size = 1
        intercept_neuron_sizes = self.hidden_layer_sizes + [output_neuron_size]
        for size in intercept_neuron_sizes:
            bias_layer = []
            for _ in range(size):
                bias_layer.append(0)
            bias_layer = np.array(bias_layer)
            self.intercepts_.append(bias_layer)
    
    def _reset_errors(self):
        self.errors_ = []
        output_neuron_size = 1
        error_neuron_sizes = self.hidden_layer_sizes + [output_neuron_size]
        for size in error_neuron_sizes:
            error_layer = []
            for _ in range(size):
                error_layer.append(0)
            error_layer = np.array(error_layer)
            self.errors_.append(error_layer)
    
    def _feed_forward_phase(self, batch_data, instance_target):
        result = np.array(batch_data)
        for layer_index in range(len(self.coefs_)):
            result = np.matmul(result, self.coefs_[layer_index])
            
            # add with bias and convert all results with sigmoid function
            for result_index in range(len(result)):
                result[result_index] = [x+y for x, y in zip(result[result_index], self.intercepts_[layer_index])]
                result[result_index] = logistic.cdf(result[result_index])
        
        return result

    def _backpropagation(self, batch_data, batch_target):
        return None

    def _update_weight(self):
        return None

    def _split_data(self, data, target):
        batch_data_list = []
        batch_target_list = []
        data_copy = deepcopy(data)
        target_copy = deepcopy(target)
        while (len(data_copy) > self.batch_size):
            batch_data = data_copy[:self.batch_size]
            batch_target = target_copy[:self.batch_size]
            batch_data_list.append(batch_data)
            batch_target_list.append(batch_target)
            data_copy = data_copy[self.batch_size:]
            target_copy = target_copy[self.batch_size:]
        batch_data_list.append(data_copy)
        batch_target_list.append(target_copy)
        return batch_data_list, batch_target_list

    def fit(self, data, target):
        self._initialize_coefs(data)
        self._initialize_intercepts(data)
        batch_data_list, batch_target_list = self._split_data(data, target)
        batches = math.ceil(len(data)/self.batch_size)
        print(batch_data_list)
        print(batch_target_list)
        for batch_index in range(batches):
            self._reset_errors()
            res = self._feed_forward_phase(batch_data_list[batch_index], batch_target_list[batch_index])
            print("this is the result for batch %d" % batch_index)
            print(res)