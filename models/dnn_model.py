import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def create_cnn1d(dim1, dim2, drop_out=None):
	model = Sequential([
		layers.Conv1D(256, 11, input_shape = (dim1, dim2), kernel_initializer=initializers.glorot_uniform),
		#layers.BatchNormalization(),
		layers.Activation('relu'),
		#layers.Dropout(0.2),
		layers.Conv1D(96, 7, kernel_initializer= initializers.glorot_uniform),
		#layers.BatchNormalization(),
		layers.Activation('relu'),
		layers.Conv1D(32, 7, kernel_initializer= initializers.glorot_uniform),
		#layers.BatchNormalization(),
		layers.Activation('relu'),
		layers.GlobalAveragePooling1D(),
		layers.Dense(128, activation='relu', kernel_initializer= initializers.glorot_uniform),
		#layers.Dropout(0.2),
		layers.Dense(64, activation='relu', kernel_initializer= initializers.glorot_uniform),
		layers.Dense(1, kernel_initializer= initializers.glorot_uniform)
	])
	model.compile(loss="mse", optimizer = tf.keras.optimizer.Adam(learning_rate=0.001))
	return model

def create_cnn2d(window_length, train_set_shape):
	model = Sequential([
		layers.Conv2D(32, (5,5), input_shape = (window_length, train_set_shape[2],1), kernel_initializer=initializers.glorot_uniform),
		#layers.BatchNormalization(),
		layers.Activation('relu'),
		layers.Conv2D(32, (3,3), kernel_initializer= initializers.glorot_uniform),
		#layers.BatchNormalization(),
		layers.Activation('relu'),
		layers.MaxPooling2D(pool_size=(2,2)),
		layers.Dropout(0.2),
		layers.Conv2D(64, (3,3), kernel_initializer= initializers.glorot_uniform),
		#layers.BatchNormalization(),
		layers.Activation('relu'),
		layers.GlobalAveragePooling2D(),
		layers.Dense(128, activation='relu', kernel_initializer= initializers.glorot_uniform),
		#layers.Dropout(0.2),
		layers.Dense(64, activation='relu', kernel_initializer= initializers.glorot_uniform),
		layers.Dense(1, kernel_initializer= initializers.glorot_uniform)
	])
	model.compile(loss="mse", optimizer = tf.keras.optimizer.Adam(learning_rate=0.001))
	return model

def create_lstm(dim2, dim3, drop_out):
	model = Sequential([
		layers.LSTM(128, input_shape = [dim2, dim3], return_sequences=True, kernel_initializer= initializers.glorot_uniform), #activation = "tanh",
		# layers.LSTM(128, return_sequences=True, kernel_initializer= initializers.glorot_uniform), #activation = "tanh",
		layers.LSTM(256, return_sequences=True, kernel_initializer= initializers.glorot_uniform),
		layers.LSTM(512, return_sequences=False, kernel_initializer= initializers.glorot_uniform),
		layers.Dense(96, activation='relu', kernel_initializer= initializers.glorot_uniform),
		layers.Dense(128, activation='relu', kernel_initializer= initializers.glorot_uniform),
		layers.Dropout(drop_out),
		#layers.Dense(64, activation='relu'),
		layers.Dense(1, kernel_initializer= initializers.glorot_uniform)
	])
	model.compile(loss="mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
	return model

def create_crnn(window_length, train_set_shape, drop_out):
	model = Sequential([
		#layers.Conv1D(32, 1, input_shape=(window_length, train_set_shape[2]), padding="same", kernel_initializer= initializers.glorot_uniform),
		# layers.BatchNormalization(),
		# layers.Activation("relu"),
		layers.Conv1D(128, 7, input_shape=(window_length, train_set_shape[2]), activation="relu",padding="same", kernel_initializer= initializers.glorot_uniform),
		# layers.BatchNormalization(),
		# layers.Activation("relu"),
		layers.MaxPooling1D(pool_size=2),
		# layers.Dropout(drop_out),
		layers.Conv1D(256, 5, activation="relu",padding="same", kernel_initializer= initializers.glorot_uniform), # activation="relu"
		# layers.BatchNormalization(),
		# layers.Activation("relu"),
		layers.MaxPooling1D(pool_size=2),
		# layers.Dropout(drop_out),
		layers.SimpleRNN(64, return_sequences=True, kernel_initializer= initializers.glorot_uniform),
		# layers.SimpleRNN(96, return_sequences=True, kernel_initializer= initializers.glorot_uniform),
		layers.SimpleRNN(128, return_sequences=True, kernel_initializer= initializers.glorot_uniform),
		# layers.Dense(64, activation="relu"),
		# layers.Dropout(drop_out),
		layers.Dense(8)

	])
	model.compile(loss="mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
	print(model.summary())
	return model

def create_clstm(window_length, train_set_shape, drop_out):
	model = Sequential([
		layers.Conv1D(64, 7, input_shape=(window_length, train_set_shape[2]), activation="relu",padding="same", kernel_initializer= initializers.glorot_uniform),
		layers.MaxPooling1D(pool_size=2),
		layers.Conv1D(128, 5, activation="relu",padding="same", kernel_initializer= initializers.glorot_uniform),
		layers.MaxPooling1D(pool_size=2),
		# layer.Dropout(0.15),
		layers.LSTM(128, return_sequences=True, kernel_initializer= initializers.glorot_uniform),
		layers.LSTM(64, return_sequences=False, kernel_initializer= initializers.glorot_uniform),
		# layers.Dense(64, activation= "relu"),
		layers.Dense(1, kernel_initializer= initializers.glorot_uniform)
	])
	model.compile(loss="mse", optimizer = tf.keras.optimizer.Adam(learning_rate=0.001))
	return model

def simple_dnn(dim1, dim2):
	model = Sequential([
		layers.Dense(40, input_shape=(dim1, dim2), activation='relu'),
		layers.Dense(20, activation='relu'),
		layers.Dense(1)
	])
	model.compile(loss="mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
	return model

def linear():
	linear = tf.keras.Sequential([
		tf.keras.layers.Dense(units=1)
	])
	linear.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.MeanAbsoluteError()])
	return linear

def dense():
	dense = tf.keras.Sequential([
		tf.keras.layers.Dense(units=64, activation = 'relu'),
		tf.keras.layers.Dense(units=64, activation = 'relu'),
		tf.keras.layers.Dense(units=1)
	])
	dense.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.MeanAbsoluteError()])
	return dense

def multi_step_dense():
	multi_step_dense = tf.keras.Sequential([
		# Shape: (time, features) => (time*features)
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1),
		# Add back the time dimension.
		# Shape: (outputs) => (1, outputs)
		tf.keras.layers.Reshape([1,-1])

	])

def conv_model():
	conv_model = tf.keras.Sequential([
		tf.keras.layers.Conv1D(filters=32,
							kernel_size=(3, ),
							activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1),
	])
	conv_model.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.MeanAbsoluteError()])
	return conv_model

output_steps = 744
num_features = 12

def multi_linear_model():
	multi_linear_model = tf.keras.Sequential([
		tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
		tf.keras.layers.Dense(output_steps * num_features,
							  kernel_initializer=tf.initializers.zeros()),
		tf.keras.layers.Reshape([744, 12])
	])
	multi_linear_model.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.MeanAbsoluteError()])
	return multi_linear_model

def multi_dense_model():
	multi_dense_model = tf.keras.Sequential([
		# Take the last time step.
		# Shape [batch, time, features] => [batch, 1, features]
		tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
		# Shape => [batch, 1, dense_units]
		tf.keras.layers.Dense(512, activation='relu'),
		# Shape => [batch, out_steps*features]
		tf.keras.layers.Dense(output_steps * num_features,
							kernel_initializer=tf.initializers.zeros()),
		# Shape => [batch, out_steps, features]
		tf.keras.layers.Reshape([output_steps, num_features])
	])
	multi_dense_model.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.RootMeanSquaredError()])
	return multi_dense_model

def multi_conv_model():
	conv_width = 24
	multi_conv_model = tf.keras.Sequential([
		tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
		tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(conv_width)),
		tf.keras.layers.Dense(output_steps*num_features, kernel_initializer=tf.initializers.zeros()),
		tf.keras.layers.Reshape([output_steps, num_features])
	])
	multi_conv_model.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.RootMeanSquaredError()])
	return multi_conv_model

def multi_lstm_model():
	multi_lstm_model = tf.keras.Sequential([
		tf.keras.layers.LSTM(32, return_sequences=True),
		tf.keras.layers.LSTM(64, return_sequences=False),
		tf.keras.layers.Dense(output_steps*num_features, kernel_initializer=tf.initializers.zeros()),
		tf.keras.layers.Reshape([output_steps, num_features])
	])
	multi_lstm_model.compile(loss=tf.losses.MeanSquaredError(),
				   optimizer=tf.optimizers.Adam(),
				   metrics=[tf.metrics.RootMeanSquaredError()])
	return multi_lstm_model
