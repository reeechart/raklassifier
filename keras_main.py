from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split

data_tennis = pd.read_csv("tennis.csv", dtype="category")

outlook_dict = {"sunny":0, "overcast":1, "rainy":2 }
windy_dict = {"FALSE": 0, "TRUE":1}
play_dict = {"no":0, "yes":1}

data_tennis["outlook"] = data_tennis["outlook"].cat.codes
data_tennis["windy"] = data_tennis["windy"].cat.codes
data_tennis["play"] = data_tennis["play"].cat.codes
 
target_tennis = data_tennis.play
data_tennis = data_tennis.drop("play", axis=1)

data_tennis = data_tennis.values
target_tennis = target_tennis.values


data_train, data_test, target_train, target_test = train_test_split(
    data_tennis, target_tennis, test_size=0.1)

model = Sequential()

model.add(Dense(8, input_dim=4, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data_train, target_train, epochs=10, batch_size=1)
loss_and_metrics = model.evaluate(data_test, target_test, batch_size=2)