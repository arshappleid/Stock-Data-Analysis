from numpy import mean
from sklearn.linear_model import LinearRegression
import pandas as pd;


class neural_net:
	
	def __init__(self,train_data , latest_observation):
		self.train_data = train_data;   ## Ideally all the data , uptill we have the corressponding closing price till.
		self.latest_observation = latest_observation;
		self.update_models();

	def getPredictedValue(self):
		observation = self.latest_observation;
		prediction1 = self.mlr_Model.predict([[observation[0], observation[1], observation[2]]])
		prediction2 = self.timeSeries_Model.predict(1); ## Get the next predicted observation
		
		return mean([prediction1 , prediction2]);
	
	def update_models(self):
		self.mlr_Model = self.multiple_linear_regression_prediction();
		self.timeSeries_Model = self.time_series_forecasting_prediction();

	def multiple_linear_regression_prediction(self):
		"""
		Pass this method the data , and it will return the y-predicted close value for the latest data, This will rely on the n-1 to train the model.
	
      	Also since we have already tested the variables , if they successfully help predict the Close price , we do not need to report the numbers.
	
      	Ideally train data can be all the stock data we have up until the latest date , and then we can use the model to predict Todays Closing price.
		"""
		train_data = self.train_data;
		linear_reg_data = train_data[['Date','Open', 'High', 'Low','Close']];
		linear_reg_data['Date'] = pd.to_datetime(linear_reg_data['Date'], errors='coerce')  # Add 'errors' parameter to handle any incorrect date formats
		linear_reg_data = linear_reg_data.dropna(subset=['Date'])
		# Convert datetime objects to ordinal
		linear_reg_data['Date'] = linear_reg_data['Date'].apply(lambda x: x.toordinal())
		# Assume we have 'Open', 'High', 'Low', 'Volume' columns and we want to predict 'Close' price.
		x = linear_reg_data[['Open', 'High', 'Low']]
		y = linear_reg_data['Close']

		# Create a Linear Regression object
		regressor = LinearRegression()
		# Train the model using the training sets
		regressor.fit(x, y)
		# Make predictions using the testing set
		return regressor;

	def time_series_forecasting_prediction(self):
		# Predict for the next 'n' days
		data = self.train_data;
		forecast_out = 1

		# Create another column (the target) shifted 'n' units up
		data['Prediction'] = data[['Close']].shift(-forecast_out)

		# Create the independent and dependent data sets
		X = data.drop(['Prediction'], 1)[:-forecast_out]
		y = data['Prediction'][:-forecast_out]

		x_train = data[['Open','High','Low']];
		y_train = data[['Close']];

		# Create and train the Linear Regression Model
		lr = LinearRegression()
		lr.fit(x_train, y_train)
		return lr;

	


	
	