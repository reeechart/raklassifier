from classifier.feed_forward_neural_network import FeedForwardNeuralNetwork

from sklearn import datasets

iris = datasets.load_iris()

clf_ffnn = FeedForwardNeuralNetwork(learning_rate=0.4, hidden_layer_sizes=[5,2], batch_size=len(iris.data))
print(clf_ffnn.learning_rate)
print(clf_ffnn.hidden_layer_sizes)
print(clf_ffnn.tol)
print(clf_ffnn.batch_size)
print(clf_ffnn.momentum)

clf_ffnn.fit(iris.data, iris.target)
print(clf_ffnn.coefs_)
print(clf_ffnn.intercepts_)
print('this is gradient')
print(clf_ffnn.gradients_)