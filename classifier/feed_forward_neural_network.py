import math
import numpy as np
from scipy.stats import logistic
from copy import deepcopy

class FeedForwardNeuralNetwork:
    def __init__(self, learning_rate, hidden_layer_sizes, max_iter=300, tol=0.0001, batch_size=1, momentum=0):
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.momentum = momentum
        self._check_validity()
        self.coefs_ = []
        self.intercepts_ = []
        self.gradients_ = []
        self.delta_gradients_ = []
        self.delta_coefs_ = []
        self.delta_intercepts_ =[]
        self.layers_input = []
        self.error_ = 0
        self.result_ = 0

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
                    neuron_weight.append(0.5)
                layer_weight.append(neuron_weight)
            layer_weight = np.array(layer_weight)
            self.coefs_.append(layer_weight)
            self.delta_coefs_.append(layer_weight)

    def _initialize_intercepts(self, data):
        output_neuron_size = 1
        intercept_neuron_sizes = self.hidden_layer_sizes + [output_neuron_size]
        for size in intercept_neuron_sizes:
            bias_layer = []
            for _ in range(size):
                bias_layer.append(0.5)
            bias_layer = np.array(bias_layer)
            self.intercepts_.append(bias_layer)
            self.delta_intercepts_.append(bias_layer)
    
    def _initialize_gradients(self):
        self.gradients_ = []
        output_neuron_size = 1
        neuron_gradient_sizes = self.hidden_layer_sizes + [output_neuron_size]
        for layer_size in neuron_gradient_sizes:
            layer_gradient = []
            for _ in range(layer_size):
                layer_gradient.append(0)
            self.gradients_.append(np.array(layer_gradient))

    def _compress_delta_gradient(self):
        compressed_delta_gradients = []
        for index in range(len(self.delta_gradients_)):
            compressed_delta_gradients.append(np.sum(self.delta_gradients_[index], axis=0))

        return compressed_delta_gradients
    
    def _feed_forward_phase(self, batch_data, batch_target):
        self.layers_input = []
        self.delta_gradients_ = []
        self.layers_input.append(np.array(batch_data))
        self.result_ = np.array(batch_data)
        for layer_index in range(len(self.coefs_)):
            self.result_ = np.matmul(self.result_, self.coefs_[layer_index])

            # add with bias and convert all results with sigmoid function
            for result_index in range(len(self.result_)):
                self.result_[result_index] = [x+y for x, y in zip(self.result_[result_index], self.intercepts_[layer_index])]
                self.result_[result_index] = logistic.cdf(self.result_[result_index])

            # save to delta_gradients
            self.delta_gradients_.append(deepcopy(self.result_))
            if (layer_index != (len(self.coefs_)-1)):
                self.layers_input.append(deepcopy(self.result_))

        diff = self.result_.ravel() - batch_target
        self.error_ += np.sum(0.5 * diff * diff)
        
        return self.result_.ravel()

    def _backpropagation(self, batch_target, batch_result):
        dg_length = len(self.delta_gradients_)
        for i in range(dg_length):
            if (i == 0):
                # output layer
                diff = batch_target - batch_result
                self.delta_gradients_[-1] = self.delta_gradients_[-1] * (1 - self.delta_gradients_[-1]) * np.reshape(diff, (len(diff), 1))
            else:
                # hidden layers
                sum_outputs_gradients = np.matmul(self.delta_gradients_[dg_length-i], np.transpose(self.coefs_[dg_length-i]))
                self.delta_gradients_[dg_length-i-1] = self.delta_gradients_[dg_length-i-1] * (1 - self.delta_gradients_[dg_length-i-1]) * sum_outputs_gradients

    def _update_weight(self, batch_data, batch_target):
        compressed_delta_input = []
        for i in range(len(self.delta_gradients_)):
            compressed_delta_input.append(np.matmul(np.transpose(self.layers_input[i]), self.delta_gradients_[i]))

        compressed_delta_input = np.array(compressed_delta_input)
        for i in range(len(self.delta_coefs_)):
            self.delta_coefs_[i] = self.learning_rate * compressed_delta_input[i] + self.momentum * self.delta_coefs_[i]
            self.coefs_[i] = np.add(self.coefs_[i], self.delta_coefs_[i])

        self.gradients_ = self._compress_delta_gradient()
        for i in range(len(self.delta_intercepts_)):
            self.delta_intercepts_[i] = self.learning_rate * self.gradients_[i] + self.momentum * self.delta_intercepts_[i]
            self.intercepts_[i] = np.add(self.intercepts_[i], self.delta_intercepts_[i])

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
        self._initialize_gradients()
        batch_data_list, batch_target_list = self._split_data(data, target)
        batches = math.ceil(len(data)/self.batch_size)

        # epochs started
        iter = 0
        while ((iter < self.max_iter and self.error_ > self.tol) or (iter == 0)):
            self.error_ = 0
            for batch_index in range(batches):
                res = self._feed_forward_phase(batch_data_list[batch_index], batch_target_list[batch_index])
                self._backpropagation(batch_target_list[batch_index], res)
                self._update_weight(batch_data_list[batch_index], batch_target_list[batch_index])
            iter += 1

    def predict(self, data):
        result = np.array(data)
        result = np.atleast_2d(result)
        for layer_index in range(len(self.coefs_)):
            result = np.matmul(result, self.coefs_[layer_index])

            # add with bias and convert all results with sigmoid function
            for result_index in range(len(result)):
                result[result_index] = [x+y for x, y in zip(result[result_index], self.intercepts_[layer_index])]
                result[result_index] = logistic.cdf(result[result_index])
        
        return result
