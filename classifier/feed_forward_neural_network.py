class FeedForwardNeuralNetwork:
    def __init__(self, learning_rate, hidden_layer_sizes, tol=0.0001, batch_size=1, momentum=0):
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.tol = tol
        self.batch_size = batch_size
        self.momentum = momentum
        self._check_validity()
        self.coefs_ = []

    def _initialize_coefs(self):
        print('initializing coefs...')

    def _initialize_intercepts(self):
        print('initializing intercepts...')

    def _check_validity(self):
        if (not self._hidden_layer_sizes_valid()):
            raise(AttributeError('Hidden layer sizes %s is invalid' % (self.hidden_layer_sizes,)))
    
    def _hidden_layer_sizes_valid(self):
        _valid_length = (len(self.hidden_layer_sizes) in range (0, 11))
        _valid_content = all(x > 0 for x in self.hidden_layer_sizes)
        return _valid_length and _valid_content
    
    def _feed_forward_phase(self, data):
        return None

    def _backpropagation(self):
        return None

    def _update_weight(self):
        return None

    def fit(self, data):
        print('fitting...')