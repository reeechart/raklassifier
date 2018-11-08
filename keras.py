from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

data_tennis = pd.read_csv("tennis.csv", dtype="category")

outlook_dict = {"sunny":0, "overcast":1, "rainy":2 }
temp_dict = {"hot":0, "mild":1, "cool":2}
humidity_dict = {"high":0, "normal":1}
windy_dict = {"False": 0, "True":1}
play_dict = {"no":0, "yes":1}

# Encode string category to integer
for column in data_tennis:
    data_tennis[column] = data_tennis[column].cat.codes

    
target_tennis = data_tennis.play
data_tennis = data_tennis.drop("play", axis=1)

data_tennis = data_tennis.values
target_tennis = target_tennis.values

model = Sequential()

model.add(Dense(8, input_dim=4, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(data_tennis, target_tennis, epochs=10, batch_size=15)