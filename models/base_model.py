import tensorflow as tf
from tensorflow import keras
import yaml
import os

class baseModel():

	def __init__(self, run_path=''):

		with open("./config.yaml") as f:
			conf = yaml.safe_load(f)
			self.input_length = conf['build_data']['input_length']
			self.epochs = conf['build_model']['epochs']
			self.batch_size = conf['build_model']['batch_size']
			self.lr = conf['build_model']['lr']
			self.decay_steps = conf['build_model']['decay_steps']
			self.decay_rate = conf['build_model']['decay_rate']
			self.history_dir = conf['output_files']['history_dir']
			self.log_dir = conf['output_files']['log_dir']
			self.drop_out = conf['build_model']['drop_out']
			self.model_dir = conf['saved_model']['model_dir']
		f.close()

		model_path = self.history_dir + run_path + '/checkpoint_{epoch:02d}-{val_loss:.2f}.hdf5'
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir + run_path)

		self.callback_lr = tf.keras.callbacks.LearningRateScheduler( tf.keras.optimizers.schedules.ExponentialDecay(
			self.lr, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True), verbose=1)
		self.callback_tb = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir + run_path)
		self.callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
			save_freq='epoch', options=None)
		self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
															   patience=2,
															   mode='min')
	
	def build(self, model):
		built_model = model()
		return built_model
	
	def train(self, built_model, train_ds, valid_ds):
		history = built_model.fit(train_ds, epochs=self.epochs,
								  validation_data=valid_ds,
								  callbacks=[self.early_stopping, self.callback_lr, self.callback_tb, self.callback_checkpoint], 
								  verbose=2)
		return history
	
	def load_model(self):
		model = keras.models.load_model(self.model_dir)
		return model
	
	def predict(self, built_model, test_set):
		return built_model.predict(test_set)