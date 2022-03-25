from models.run_model import runModel

if __name__ == '__main__':
	model = runModel()
	mode = model.mode #check if it is train or test
	model.load_data() #load the input data and apply the required preprocessing steps

	if mode == 'train':
		model.train() #train the specified model using train dataset
	elif mode == 'test':
		model.test() #create the prediction for test dataset using the saved model
	elif mode == 'eda':
		pass