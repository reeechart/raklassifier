from classifier.feed_forward_neural_network import FeedForwardNeuralNetwork

from sklearn import datasets

iris = datasets.load_iris()

clf_ffnn = FeedForwardNeuralNetwork(learning_rate=0.4, hidden_layer_sizes=[7,6,5,4,4,3,2,1,1,1])
print(clf_ffnn.learning_rate)
print(clf_ffnn.hidden_layer_sizes)
print(clf_ffnn.tol)
print(clf_ffnn.batch_size)
print(clf_ffnn.momentum)

clf_ffnn.fit(iris.data)
print(clf_ffnn.coefs_)