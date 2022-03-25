from models.base_model import baseModel
from utils.data_reader import dataReader, WindowGenerator
from models.dnn_model import create_cnn1d, create_cnn2d, create_lstm, create_crnn, create_clstm, linear, dense, conv_model, multi_linear_model, multi_dense_model, multi_conv_model, multi_lstm_model
from utils.eda import save_results, save_predicted_test, create_heatmap, visualize_weights, plot_prediction, delete_negatives, plot_hist2d, wind_vector
import os
import yaml
from keras.backend import clear_session
from shutil import copyfile
import numpy as np

class runModel(baseModel, dataReader, WindowGenerator):

	#reading input parameters including data directories and hyperparameters from the config file
	def __init__(self):
		welcome_log = 'Thank you for using this sequence modeling tool to predict energy generation!'
		print(welcome_log)
		conf_file = "./config.yaml"
		with open(conf_file) as f:
			conf = yaml.safe_load(f)
			self.mode = conf['mode']
			self.run_num = conf['run_num']
			self.model_name = conf['model_name']
			self.history_dir = conf['output_files']['history_dir']
			self.pred_file = conf['saved_model']['pred_file']
			self.smooth = conf['build_data']['denoising']
			self.columns_for_MA = conf['build_data']['columns_for_MA']
			self.model_dict = {'CNN1D': create_cnn1d, 'CNN2D': create_cnn2d, 'LSTM': create_lstm, 'CRNN': create_crnn, 'CLSTM': create_clstm, 'Linear': linear, 'Dense' : dense, 'Conv_model':conv_model, 'Multi_linear_model':multi_linear_model, 'Multi_dense_model':multi_dense_model, 'Multi_conv_model':multi_conv_model, 'Multi_lstm_model':multi_lstm_model}
			self.input_length = conf['build_data']['input_length']
			self.label_length = conf['build_data']['label_length']
			self.shift = conf['build_data']['shift']
		f.close()

	# this function reads the train and test datasets from the directory given in the config file, then extract the train labels 
	# and applies a series of preprocessing methods such as feature reduction, noise removing, feature scaling and feature selection
	# and finally create batches for train and test dataset to feed them to the network

	def load_data(self):
		dataset = dataReader()
		generation_dataset = dataset.get_generation() #reading generation dataset.  this is y.
		temperature_dataset = dataset.get_temperature() #reading temperature dataset.  these are x's.
		data = temperature_dataset.merge(generation_dataset, how='left', on='DateTime') #mergind two dataframes to create train data
		data = dataset.convert_index_int(data) #changing date index to integers
		data = dataset.value_time(data) #creating frequency of day and year

		data = dataset.imputation(data) # filling in missing values in WWCode
		dataset.convert_val_to_zero(data, 'WWCode') # some values become negative after imputation.  converting them to 0
		
		# Creating wind vector by using wind speed and direction
		data = wind_vector(data)


		if self.smooth:
			for col in self.columns_for_MA:
				data = dataset.smooth(data, col) # taking moving averages.  creates new columns for MA's.

		data = dataset.drop_col(data)

		feature_columns = data.columns
		# print(feature_columns)
		
		#Plotting 2D histogram for wind direction and wind speed
		# plot_hist2d(data)

		train_df, valid_df, test_df = dataset.create_train_valid_test_sets(data) # output is dataframes
		self.train_df = train_df # for visualization in the train function below
		scaled_train, scaled_valid, scaled_test, self.scaler = dataset.feature_scaler(train_df, valid_df, test_df, feature_columns) # output is dataframes
		self.scaled_test_df = scaled_test # to predict in the test function below

		# print('Scaled train:')
		# print(scaled_train.describe())

		# Creating datasets
		self.w1 = WindowGenerator(self.input_length, self.label_length, self.shift, scaled_train, scaled_valid, scaled_test, label_columns=['Generation'])
		
		# print(self.w1)
		# self.w1.plot(plot_col='Generation')
		# print('Shape of train dataset: {}'.format(len(list(self.w1.train.as_numpy_iterator()))))
		# print('Shape of valid dataset: {}'.format(len(list(self.w1.val.as_numpy_iterator()))))
		# print('Shape of test dataset: {}'.format(len(list(self.w1.test.as_numpy_iterator()))))
		# print('Train dataset# 0: {}'.format(list(self.w1.train.as_numpy_iterator())[0][0].shape))
		# print('Valid dataset# 0: {}'.format(list(self.w1.val.as_numpy_iterator())[0][0].shape))
		# print('Test dataset# 0: {}'.format(list(self.w1.test.as_numpy_iterator())[0][0].shape))
		# print('Test dataset label# 0: {}'.format(list(self.w1.test.as_numpy_iterator())[0][1].shape))
				
		#--------------------------------------------------
		# EXPLANATORY DATA ANALYSIS
		#--------------------------------------------------
		# Plotting each column
		# dataset.plot_features(data)

		# Creating heatmap to check colinearity
		# create_heatmap(data)
	
		# dataset.plot_feature_MA(data, ['AirTemperature', 'AirTemperature_MA'])
		# dataset.plot_feature_MA(data, ['RelativeHumidity', 'RelativeHumidity_MA'])

		#--------------------------------------------------

		
		
	# this function selects and builds the neural network model based on the config file,
	# then train the model using the train batches created in the previous step and based on the parameters given in the config file,
	# and finally save the best checkpoints and model configuration in the given directory

	def train(self):
		model_func = self.model_dict[self.model_name]
		
		if not os.path.exists(self.history_dir):
			os.makedirs(self.history_dir)
		run_path = '/{:02d}'.format(self.run_num)
		if not os.path.exists(self.history_dir + run_path):
			os.makedirs(self.history_dir + run_path)
		
		base = baseModel(run_path=run_path)
		model = base.build(model_func)
		
		history = base.train(model, self.w1.train, self.w1.val)

		save_results(history, self.history_dir + run_path, self.model_name)
		with open(self.history_dir + run_path + '/{}_model_summary.txt'.format(self.model_name), 'w') as f:
			model.summary(print_fn=lambda x: f.write(x + '\n'))
		copyfile("./config.yaml",self.history_dir + run_path + '/{}_conf.yaml'.format(self.model_name))

		print('Train {} is complete!'.format(self.run_num))
		
		# self.w1.plot(model=model, plot_col='Generation')

		# visualize_weights(model, self.train_df)


		clear_session()
		

	
	# this function loads the trained model (including model architecture and weights) from the given directory in the config file, 
	# and predict the remaining useful life (RUL) for the test dataset and report the root mean square error (RMSE)
	def test(self):
		base = baseModel()
		model = base.load_model()
		scaled_test_np = self.scaled_test_df.to_numpy()
		scaled_test_pred = model.predict(np.expand_dims(scaled_test_np,0))
		# print(scaled_test_pred.shape)
		scaled_test_pred_inv = self.scaler.inverse_transform(scaled_test_pred[0, :, :])
		# print(scaled_test_pred)
		test_pred = delete_negatives(scaled_test_pred_inv[:, 3]) #converting negatives to 0
		# plot_prediction(test_pred)
		save_predicted_test(test_pred, self.pred_file + "/submission1_.csv")
		print('Prediction result has been saved.')
