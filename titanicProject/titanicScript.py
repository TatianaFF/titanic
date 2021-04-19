import numpy as np
import tensorflow as tf
import keras as k
import numpy
import pandas
import matplotlib.pyplot as plt

# #обучение нс
# input_data = numpy.array([0.3,0.7,0.9])
# output_data = numpy.array([0.5,0.9,1.0])
# 
# model = k.Sequential()
# model.add(k.layers.Dense(units=1,activation='linear'))
# model.compile(loss='mse', optimizer='sgd')
# fit_results = model.fit(x=input_data, y=output_data, epochs=100)
#
#проверка качества обучения
#предсказание рез-та
# predicted = model.predict([0.5])
# print(predicted)

# подготовка данных
data_frame = pandas.read_csv("titanic.csv")
input_names = ["Age", "Sex", "Pclass"]
output_names = ["Survived"]


# сырые данные
raw_input_data = data_frame[input_names]
raw_output_data = data_frame[output_names]

max_age = 100
encoders = {"Age": lambda age: [age / max_age],
            "Sex": lambda gen: {"male": [0], "female": [1]}.get(gen),
            "Pclass": lambda pclass: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(pclass),
            "Survived": lambda s_value: [s_value]}


# функция для перевода из csv в словарь
def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result


# перевод из csv в словарь
def make_supervised(df):
    return {
        "inputs": dataframe_to_dict(raw_input_data),
        "outputs": dataframe_to_dict(raw_output_data)
    }


# кодирование данных
def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    # сопоставление значений по векторам
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


# на выходе вектора входные [0.22, 0, [0, 1, 0]], выходные [0]


supervised = make_supervised(data_frame)
encoded_inputs = numpy.array(encode(supervised["inputs"]))
encoded_outputs = numpy.array(encode(supervised["outputs"]))

train_x = encoded_inputs[:600]
train_y = encoded_outputs[:600]

test_x = encoded_inputs[600:]
test_y = encoded_outputs[600:]

model = k.Sequential()
model.add(k.layers.Dense(units=5, activation="relu"))  # act linear
model.add(k.layers.Dense(units=1, activation="sigmoid"))
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
fit_results = model.fit(x=train_x, y=train_y, epochs=100, validation_split=0.2)
# loss тренировочные потери, val_loss валидационная потеря

plt.title("Losses train/validation")
plt.plot(fit_results.history["loss"], label="Train")
plt.plot(fit_results.history["val_loss"], label="Validation")
plt.legend()
plt.show()


plt.title("Accuracy train/validation")
plt.plot(fit_results.history["accuracy"], label="Train")
plt.plot(fit_results.history["val_accuracy"], label="Validation")
plt.legend()
plt.show()

predicted_test = model.predict(test_x)
real_data = data_frame.iloc[600:][input_names+output_names]
real_data["PSurvived"] = predicted_test
print(real_data)