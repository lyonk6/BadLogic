import numpy as np
import keras

def generate_time_series(batch_size, steps):
    f1, f2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, steps)
    series  = 0.5 * np.sin((time - offset1) * (f1 * 10 + 10)) # first wave
    series += 0.2 * np.sin((time - offset2) * (f2 * 20 + 20)) # second wave
    series += 0.1 * (np.random.rand(batch_size, steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps +1)
x_training,     y_training = series[:7000, :n_steps],      series[:7000, -1]
x_validation, y_validation = series[7000:9000, :n_steps],  series[7000:9000, -1]
x_test,             y_test = series[9000:, :n_steps],      series[9000:, -1]


y_prediction = x_validation[:, -1]


mse_prediction = np.mean(keras.losses.mean_squared_error(y_validation, y_prediction))
print("This is what we predicted from the MSE:", mse_prediction)


###################################################################################
# Feeed Forward Neural Network
###################################################################################
flat_model     = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50,1]),
    keras.layers.Dense(1)
])
flat_model.compile(loss="mse", optimizer="adam")
flat_model_history = flat_model.fit(x_training, y_training, epochs=20,
                    validation_data=(x_validation, y_validation))
flat_model_evalutation = flat_model.evaluate(x_validation, y_validation)
print("flat_model evalutation:", flat_model_evalutation)
###################################################################################


###################################################################################
# Deep Recurrent Neural Network
###################################################################################
RNN_model     = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])
###################################################################################
