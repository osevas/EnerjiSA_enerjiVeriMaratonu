import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def change_comma_to_point(df):
	for col in df.select_dtypes(include='object').columns.values:
		df[col] = df[col].str.replace(',', '.')
	return df

def dropna_in_first_col(df):
	index_to_drop = df[df['DateTime'].isna()].index
	df.drop(index=index_to_drop, inplace=True)
	return df


def min_max_scaler_all(train_data, valid_data, test_data):
	scaler = MinMaxScaler(feature_range=(0, 1))
	train_data = scaler.fit_transform(train_data)
	valid_data = scaler.transform(valid_data)
	test_data = scaler.transform(test_data)
	return train_data, valid_data, test_data, scaler

def standard_scalar_all(train_data, test_data):
	scaler = StandardScaler().fit(train_data)
	train_data = scaler.transform(train_data)
	test_data = scaler.transform(test_data)
	return train_data, test_data

def minmax_scalar_cluster(train_data, test_data, train_ops, test_ops, num_clusters=6):
	train_cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(train_ops)
	train_k = train_cluster.labels_
	test_k = train_cluster.predict(test_ops)

	for i in range(num_clusters):
		trn_idx = np.where(train_k==i)
		tst_idx = np.where(test_k == i)
		trn_temp = train_data.loc[trn_idx]
		tst_temp = test_data.loc[tst_idx]
		#scaler = MinMaxScaler(feature_range=(-1, 1)).fit(trn_temp)
		#trn_norm = scaler.transform(trn_temp)
		#tst_norm = scaler.transform(tst_temp)
		scaler = MinMaxScaler(feature_range=(-1, 1))
		trn_norm = scaler.fit_transform(trn_temp)
		tst_norm = scaler.transform(tst_temp)
		print("min_trn:", np.min(trn_norm))
		print('max_trn:', np.max(trn_norm))
		print('min_test:', np.min(tst_norm))
		print('max_test:', np.max(tst_norm))
		train_data.iloc[trn_idx[0]] = trn_norm[:]
		test_data.iloc[tst_idx[0]] = tst_norm[:]
		del trn_idx, tst_idx, trn_temp, tst_temp, scaler, trn_norm, tst_norm
	return train_data, test_data

def standard_scalar_cluster(train_data, test_data, train_ops, test_ops, num_clusters=6):
	train_cluster = KMeans(n_clusters=num_clusters, random_state=0).fit(train_ops)
	train_k = train_cluster.labels_
	test_k = train_cluster.predict(test_ops)

	for i in range(num_clusters):
		trn_idx = np.where(train_k==i)
		tst_idx = np.where(test_k == i)
		trn_temp = train_data.loc[trn_idx]
		tst_temp = test_data.loc[tst_idx]
		#scaler = MinMaxScaler(feature_range=(-1, 1)).fit(trn_temp)
		#trn_norm = scaler.transform(trn_temp)
		#tst_norm = scaler.transform(tst_temp)
		scaler = StandardScaler().fit(trn_temp)
		trn_norm = scaler.transform(trn_temp)
		tst_norm = scaler.transform(tst_temp)
		print("min_trn:", np.min(trn_norm))
		print('max_trn:', np.max(trn_norm))
		print('min_test:', np.min(tst_norm))
		print('max_test:', np.max(tst_norm))
		train_data.iloc[trn_idx[0]] = trn_norm[:]
		test_data.iloc[tst_idx[0]] = tst_norm[:]
		del trn_idx, tst_idx, trn_temp, tst_temp, scaler, trn_norm, tst_norm
	return train_data, test_data

def rmse_test(pred, true, num_test_windows_list):
	preds_for_each_engine = np.split(pred, np.cumsum(num_test_windows_list)[:-1])
	mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
								 for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]
	RMSE = np.sqrt(mean_square_error(true, mean_pred_for_each_engine))
	return RMSE

def save_results(history, save_dir, model_name):
	epoch_log = np.expand_dims(np.array(history.epoch), axis=1)
	train_loss_log = np.expand_dims(np.array(history.history["loss"]), axis=1)
	valid_loss_log = np.expand_dims(np.array(history.history["val_loss"]), axis=1)
	log_df = pd.DataFrame(np.concatenate((epoch_log, train_loss_log, valid_loss_log), axis=1),
						  columns=["epoch", "train_loss", "valid_loss"])
	log_df.to_csv(save_dir +"/{}.csv".format(model_name))
	return

def save_predicted_test(test_pred, save_dir):
	result_df = pd.read_csv('./sample_submission.csv')
	result_df['Generation'] = test_pred
	result_df.to_csv(save_dir, index=False)
	return

def create_heatmap(df):
	'''
	Creates heatmap between dataframe columns to see colinearity
	'''
	fig, ax = plt.subplots(figsize = (12, 7))
	corr = df.corr()
	sns.heatmap(corr, annot = True)
	plt.show()

def visualize_weights(model, train_df):
	plt.figure(figsize=(10, 20))
	plt.bar(x = range(len(train_df.columns)),
			height=model.layers[0].kernel[:,0].numpy())
	axis = plt.gca()
	axis.set_xticks(range(len(train_df.columns)))
	_ = axis.set_xticklabels(train_df.columns, rotation=90)
	plt.show()

def plot_prediction(predictions):
	plt.figure(figsize=(10,20))
	plt.plot(predictions)
	plt.show()

def delete_negatives(predictions):
	# one_col = predictions[0, :, 3]
	
	def func(x):
		for i in range(len(x)):
			if x[i]<0:
				x[i] = 0
		return x
	
	one_col = np.apply_along_axis(func, 0, predictions)
	return one_col

def plot_hist2d(df):
	plt.figure(figsize=(10, 20))
	# plt.hist2d(df['WindDirection'], df['WindSpeed'], bins=(50, 50))
	plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50))
	plt.colorbar()
	# plt.xlabel('Wind Direction [deg]')
	# plt.ylabel('Wind Velocity [km/h]')
	plt.xlabel('Wind X [km/h]')
	plt.ylabel('Wind Y [km/h]')
	plt.show()

def wind_vector(df):
	wv = df.pop('WindSpeed')

	# Convert to radians.
	wd_rad = df.pop('WindDirection')*np.pi / 180

	# Calculate the wind x and y components.
	df['Wx'] = wv*np.cos(wd_rad)
	df['Wy'] = wv*np.sin(wd_rad)
	return df