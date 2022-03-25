from utils.eda import change_comma_to_point, dropna_in_first_col, min_max_scaler_all
from utils.batch_data import create_dataset, shuffle_train_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import os
# from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import tensorflow as tf

class dataReader():
	def __init__(self):
		with open("./config.yaml") as f:
			conf = yaml.safe_load(f)
			self.generation = conf['data_dir']['generation']
			self.temperature = conf['data_dir']['temperature']
			self.feature_scaling = conf['build_data']['feature_scaling']
			self.input_length = conf['build_data']['input_length']
			self.shift = conf['build_data']['shift']
			self.columns_to_be_dropped = conf['build_data']['columns_to_be_dropped']
			self.train_ratio = conf['build_data']['train_ratio']
			self.plot_dir = conf['eda']['plot_dir']
			self.smooth_window = conf['build_data']['smooth_window']
			self.batch_size = conf['build_data']['batch_size']
		f.close()
		# if self.early_rul == "None":
		# 	self.early_rul = None
		
	def get_generation(self):
		generation_data = pd.read_csv(self.generation, sep=';')
		generation_data = dropna_in_first_col(generation_data) #dropping rows that have NA in column DateTime
		generation_data.set_index('DateTime', inplace=True)
		generation_data = change_comma_to_point(generation_data)
		# generation_data.dropna(axis=0, inplace=True)
		generation_data = generation_data.astype('float')
		return generation_data

	def get_temperature(self):
		temperature_data = pd.read_csv(self.temperature, sep=';')
		temperature_data = dropna_in_first_col(temperature_data) #dropping rows that have NA in column DateTime
		temperature_data.set_index('DateTime', inplace=True)
		temperature_data = change_comma_to_point(temperature_data)
		# temperature_data.dropna(axis=0, inplace=True)
		temperature_data = temperature_data.astype('float')
		# print(temperature_data.head())
		return temperature_data
	
	def plot_feature(self, df, col):
		if not os.path.exists(self.plot_dir):
			os.makedirs(self.plot_dir)

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=df.index.values, y=df.loc[:,col], mode='lines'))
		fig.update_layout(title=col, xaxis_title='DateTime')
		fig.write_html(self.plot_dir+'/'+col+'.html')
		return None

	def plot_features(self, df):
		if not os.path.exists(self.plot_dir):
			os.makedirs(self.plot_dir)

		for col in df.columns.values:
			fig = go.Figure()
			fig.add_trace(go.Scatter(x=df.index.values, y=df.loc[:,col], mode='lines'))
			fig.update_layout(title=col, xaxis_title='DateTime')
			fig.write_html(self.plot_dir+'/'+col+'_original.html')
		return None
	
	def plot_feature_MA(self, df, cols):
		if not os.path.exists(self.plot_dir):
			os.makedirs(self.plot_dir)

		fig = go.Figure()
		for col in cols:
			fig.add_trace(go.Scatter(x=df.index.values, y=df.loc[:,col], mode='lines', name=col))
		fig.write_html(self.plot_dir+'/'+col+'_MA.html')
		return None

	def get_test_data(self, df):
		return df.iloc[-744:,:]

	def create_train_valid_test_sets(self, df1):
		'''
		df1: combined data
		'''
		border = int(df1.shape[0]*self.train_ratio//1)
		train = df1.iloc[:border,:]
		valid = df1.iloc[border:(-744*2),:]
		test = df1.iloc[(-744*2):-744,:]
		print('Train_df shape: {}'.format(train.shape))
		print('Valid_df shape: {}'.format(valid.shape))
		print('Test_df shape: {}'.format(test.shape))
		return (train, valid, test)
	
	def convert_to_datetime(self, df):
		df['date_time'] = pd.to_datetime(df['date_time'], format='%d%b%Y %H:%M:%S')
		return df
	
	def convert_index_int(self, df):
		df['date_time'] = df.index.values
		df = self.convert_to_datetime(df) # converting date_time to date_time format
		df.index = np.arange(df.shape[0], dtype=np.int0)
		return df
	
	def value_time(self, df):
		day = 24 * 60 * 60
		year = 365.2425 * day

		timestamp_s = df['date_time'].map(pd.Timestamp.timestamp)

		df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
		df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
		df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
		df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

		df.drop(columns='date_time', inplace=True)
		# print(df)

		# plt.figure(figsize=(10, 20))
		# plt.plot(np.array(df['Day sin'])[:25])
		# plt.plot(np.array(df['Day cos'])[:25])
		# plt.xlabel('Time [h]')
		# plt.title('Time of day signal')
		# plt.show()
		return df

	def drop_col(self, df):
		
		df_dropped_cols = df.drop(columns=self.columns_to_be_dropped)
		# valid_dropped_cols = valid_data.drop(columns=train_data.columns.values[self.columns_to_be_dropped])
		# test_dropped_cols = test_data.drop(columns=train_data.columns.values[self.columns_to_be_dropped])
		return df_dropped_cols
	
	def smooth(self, df, col):
		col_new = col + '_MA'
		df[col_new] = df[col].rolling(self.smooth_window, min_periods=1, win_type=None).mean()
		# test_data = test_data.rolling(5, min_periods=1, win_type=None).mean()
		return df
	
	def min_max_scaler(self, train_data,valid_data, test_data, feature_columns):
		scaled_train, scaled_valid, scaled_test, scaler = min_max_scaler_all(train_data, valid_data, test_data)
		scaled_train_df = pd.DataFrame(scaled_train, columns=feature_columns)
		scaled_valid_df = pd.DataFrame(scaled_valid, columns=feature_columns)
		scaled_test_df = pd.DataFrame(scaled_test, columns=feature_columns)
		return scaled_train_df, scaled_valid_df, scaled_test_df, scaler
	
	def build_train_batch(self, train_data):
		train_set, train_target_set = create_dataset(train_data, window_length=self.input_length, shift=self.shift)
		train_set, train_target_set = shuffle_train_data(train_set, train_target_set)
		# print("Processed training data shape: ", train_set.shape)
		# print("Processed training target shape: ", train_target_set.shape)
		return train_set, train_target_set

	def build_valid_batch(self, valid_data):
		valid_set, valid_target_set = create_dataset(valid_data, window_length=self.input_length, shift=self.shift)
		# print("Processed validing data shape: ", valid_set.shape)
		# print("Processed validing target shape: ", valid_target_set.shape)
		return valid_set, valid_target_set
		
	def imputation(self, df):
		'''
		Imputing missing values by using other features
		'''
		cols = ['AirTemperature','RelativeHumidity','WindSpeed','WindDirection','WWCode','EffectiveCloudCover']

		imp = IterativeImputer(max_iter=10, random_state=0)
		imp.fit(df[cols].to_numpy())
		df[cols] = imp.transform(df[cols])
		return df
	
	def convert_val_to_zero(self, df, col):
		index_to_zerod = df[df[col]<0].index
		df.loc[index_to_zerod, col] = 0
		return df

	def read_train_data(self):
		return get_train_data(self.train_dir, self.col_name)
	
	
	def min_max_cluster(self, train_data, test_data):
		scaled_train, scaled_test = minmax_scalar_cluster(train_data, test_data, self.train_ops, self.test_ops, num_clusters= self.num_clusters)
		train_data = pd.DataFrame(data=np.c_[self.train_data_first_column, scaled_train])
		test_data = pd.DataFrame(data=np.c_[self.test_data_first_column, scaled_test])
		return train_data, test_data

	def standard_scaler(self, train_data, test_data):
		scaled_train, scaled_test = standard_scalar_all(train_data, test_data)
		train_data = pd.DataFrame(data=np.c_[self.train_data_first_column, scaled_train])
		test_data = pd.DataFrame(data=np.c_[self.test_data_first_column, scaled_test])
		return train_data, test_data

	def feature_scaler(self, train_df, valid_df, test_df, feature_columns):
		if self.feature_scaling == "min_max":
			scaled_train, scaled_valid, scaled_test, scaler = self.min_max_scaler(train_df, valid_df, test_df, feature_columns)
		if self.feature_scaling == "min_max_cluster":
			train_data, test_data = self.min_max_cluster(train_data, test_data)
		if self.feature_scaling == "standard":
			train_data, test_data = self.standard_scaler(train_data, test_data)
		if self.feature_scaling == "standard_cluster":
			train_data, test_data = self.standard_cluster(train_data, test_data)
		return scaled_train, scaled_valid, scaled_test, scaler

	def buid_test_batch(self, test_data, true_rul):
		if self.mode == "train":
			self.num_test_windows=1
		num_train_machines = len(test_data[0].unique())
		test_set, test_rul, num_test_windows_list = load_test_data(num_train_machines, test_data, true_rul, self.input_length, self.shift, self.num_test_windows)
		print("Processed test data shape: ", test_set.shape)
		print("True RUL shape: ", test_rul.shape)
		return test_set, test_rul, num_test_windows_list
	
class WindowGenerator():
	def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
		# Store the raw data.
		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df

		# Work out the label column indices.
		self.label_columns = label_columns
		if label_columns is not None:
			self.label_columns_indices = {name: i for i, name in
											enumerate(label_columns)}
		self.column_indices = {name: i for i, name in
							enumerate(train_df.columns)}

		# Work out the window parameters.
		self.input_width = input_width
		self.label_width = label_width
		self.shift = shift

		self.total_window_size = input_width + shift

		self.input_slice = slice(0, input_width)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]

		self.label_start = self.total_window_size - self.label_width
		self.labels_slice = slice(self.label_start, None)
		self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

		with open("./config.yaml") as f:
			conf = yaml.safe_load(f)
			self.batch_size = conf['build_data']['batch_size']
		f.close()

	def __repr__(self):
		return '\n'.join([
			f'Total window size: {self.total_window_size}',
			f'Input indices: {self.input_indices}',
			f'Label indices: {self.label_indices}',
			f'Label column name(s): {self.label_columns}'])
	
	def split_window(self, features):
		inputs = features[:, self.input_slice, :]
		labels = features[:, self.labels_slice, :]
		if self.label_columns is not None:
			labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

		# Slicing doesn't preserve static shape information, so set the shapes
		# manually. This way the `tf.data.Datasets` are easier to inspect.
		inputs.set_shape([None, self.input_width, None])
		labels.set_shape([None, self.label_width, None])

		return inputs, labels

	def plot(self, model=None, plot_col=None, max_subplots=3):
		inputs, labels = self.example
		plt.figure(figsize=(12, 8))
		plot_col_index = self.column_indices[plot_col]
		max_n = min(max_subplots, len(inputs))
		for n in range(max_n):
			plt.subplot(max_n, 1, n+1)
			plt.ylabel(f'{plot_col}')
			plt.plot(self.input_indices, inputs[n, :, plot_col_index],
						label='Inputs', marker='.', zorder=-10)

			if self.label_columns:
				label_col_index = self.label_columns_indices.get(plot_col, None)
			else:
				label_col_index = plot_col_index

			if label_col_index is None:
				continue

			plt.scatter(self.label_indices, labels[n, :, label_col_index],
						edgecolors='k', label='Labels', c='#2ca02c', s=64)
			if model is not None:
				predictions = model(inputs)
				plt.scatter(self.label_indices, predictions[n, :, label_col_index],
							marker='X', edgecolors='k', label='Predictions',
							c='#ff7f0e', s=64)

			if n == 0:
				plt.legend()

		plt.xlabel('Time [h]')
		plt.show()

	def make_dataset(self, data):
		data = np.array(data, dtype=np.float32)
		ds = tf.keras.utils.timeseries_dataset_from_array(
			data=data,
			targets=None,
			sequence_length=self.total_window_size,
			sequence_stride=1,
			shuffle=True,
			batch_size=self.batch_size)

		ds = ds.map(self.split_window)

		return ds

	@property
	def train(self):
		return self.make_dataset(self.train_df)

	@property
	def val(self):
		return self.make_dataset(self.val_df)

	@property
	def test(self):
		return self.make_dataset(self.test_df)

	@property
	def example(self):
		"""Get and cache an example batch of `inputs, labels` for plotting."""
		result = getattr(self, '_example', None)
		if result is None:
			# No example batch was found, so get one from the `.train` dataset
			result = next(iter(self.train))
			# And cache it for next time
			self._example = result
		return result