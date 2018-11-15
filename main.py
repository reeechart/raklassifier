# from classifier.feed_forward_neural_network import FeedForwardNeuralNetwork

from classifier.feed_forward_neural_network import FeedForwardNeuralNetwork
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn import datasets

data_tennis = pd.read_csv("tennis.csv", dtype="category")

outlook_dict = {"sunny": 0, "overcast": 1, "rainy": 2}
windy_dict = {"FALSE": 0, "TRUE": 1}
play_dict = {"no": 0, "yes": 1}

data_tennis["outlook"] = data_tennis["outlook"].cat.codes
data_tennis["windy"] = data_tennis["windy"].cat.codes
data_tennis["play"] = data_tennis["play"].cat.codes
data_tennis["temp"] = pd.eval(data_tennis["temp"])
data_tennis["humidity"] = pd.eval(data_tennis["humidity"])
target_tennis = data_tennis.play
data_tennis= data_tennis.drop("play", axis=1)

X = data_tennis.values
Y = target_tennis.values

print(X,Y)

clf_ffnn = FeedForwardNeuralNetwork(learning_rate=0.25, hidden_layer_sizes=[5,2], batch_size=5, max_iter=4, momentum=0.1)
print(clf_ffnn.learning_rate)
print(clf_ffnn.hidden_layer_sizes)
print(clf_ffnn.tol)
print(clf_ffnn.batch_size)
print(clf_ffnn.momentum)

clf_ffnn.fit(X, Y)
print('last coef')
print(clf_ffnn.coefs_)
print('last intercepts')
print(clf_ffnn.intercepts_)
print(clf_ffnn.predict([[0,85,85,0]]))
# print(iris)
# clf = FFNN(learning_rate=0.1, hidden_layer_sizes=[5,2], batch_size=5, max_iter=4, momentum=0)
# clf.fit(X, Y)
# print("---LAYER---")
# for layer in clf.layers:
#     print("---COEFS---")
#     print(layer.coefs)
#     print("---Intercepts---")
#     print(layer.intercepts)
# print("---ENDLAYER---")
# print(clf.predict([[0, 0, 0, 0]]))
