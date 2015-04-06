import sys
from pandas import *
from ggplot import *
import numpy as numpy
from datetime import *
import os
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def processData():

	turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
	#turnstile_weather = pandas.read_csv('turnstile_weather_v2.csv')
	
	turnstile_weather.DATEn = pandas.to_datetime(turnstile_weather.DATEn)
	turnstile_weather.TIMEn = pandas.to_datetime(turnstile_weather.TIMEn)

	print "Source data\n"
	print turnstile_weather.dtypes
	print "\n"
	print turnstile_weather.describe()
	print "\n"
	print turnstile_weather
	print "\n"


def printDistributionHistogram():

	turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
	#turnstile_weather = pandas.read_csv('turnstile_weather_v2.csv')
	
	turnstile_weather.DATEn = pandas.to_datetime(turnstile_weather.DATEn)
	turnstile_weather.TIMEn = pandas.to_datetime(turnstile_weather.TIMEn)

	binCount = 200
	
	plt.figure()
	plt.ylim(0, 45000)
	plt.xlim(0, 10000)
	plt.title('Hourly entry distribution')
	plt.ylabel('Frequency')
	plt.xlabel('ENTRIESn_hourly')
	plt.legend(loc='upper right')
	turnstile_weather[turnstile_weather.rain == 0].ENTRIESn_hourly.hist(bins=binCount, label='No rain', alpha=0.5, color='red')
	turnstile_weather[turnstile_weather.rain == 1].ENTRIESn_hourly.hist(bins=binCount, label='Rain', alpha=0.5, color='blue')
	plt.show()

def printSubwayWeatherStatistics():

	turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
	#turnstile_weather = pandas.read_csv('turnstile_weather_v2.csv')
	
	turnstile_weather.DATEn = pandas.to_datetime(turnstile_weather.DATEn)
	turnstile_weather.TIMEn = pandas.to_datetime(turnstile_weather.TIMEn)

	rainDf = turnstile_weather[turnstile_weather.rain == 1]
	altDf = turnstile_weather[turnstile_weather.rain == 0]
	with_rain_mean = np.mean(rainDf.ENTRIESn_hourly)
	without_rain_mean = np.mean(altDf.ENTRIESn_hourly)
	U, p = scipy.stats.mannwhitneyu(rainDf.ENTRIESn_hourly, altDf.ENTRIESn_hourly)
    
	print "Average entries w/ rain: " + str(with_rain_mean)
	print "Average entries w/o rain: " + str(without_rain_mean)
	print "Difference: " + str(with_rain_mean / without_rain_mean - 1)
	print "U-statistic: " + str(U)
	print "p-value: " + str(p)

	print "\nWith rain"
	print rainDf.describe()
	
	print "\nWithout rain"
	print altDf.describe()

	
def getDataPointColor(data):
	if data.rain > 0:
		return '#0000FF'
	else:
		return '#FF0000'


def getDataPointIdentifier(data):
	if data.UNIT not in unitList:
		unitList.append(data.UNIT)
	return unitList.index(data.UNIT)

	
def printTurnstileVisualization():
	turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
	#turnstile_weather = pandas.read_csv('turnstile_weather_v2.csv')
	
	turnstile_weather.DATEn = pandas.to_datetime(turnstile_weather.DATEn)
	turnstile_weather.TIMEn = pandas.to_datetime(turnstile_weather.TIMEn)

	globals()["unitList"] = list()

	turnstileGroup = turnstile_weather[['UNIT', 'DATEn', 'ENTRIESn_hourly', 'rain']].groupby(['UNIT', 'DATEn'], as_index=False).sum()
	turnstileGroup['color'] = turnstileGroup.apply(getDataPointColor, axis=1)
	turnstileGroup['turnstile'] = turnstileGroup.apply(getDataPointIdentifier, axis=1)
	
	print turnstileGroup
	
	print (ggplot(turnstileGroup, aes(x = turnstileGroup.turnstile, y = turnstileGroup.ENTRIESn_hourly)) +
		geom_point(aes(color = turnstileGroup.color)) + xlab('Turnstile') + ylab('Entries') + ggtitle("Daily entries per turnstile"))

def printDateVisualization():
	turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
	#turnstile_weather = pandas.read_csv('turnstile_weather_v2.csv')
	
	turnstile_weather.DATEn = pandas.to_datetime(turnstile_weather.DATEn)
	turnstile_weather.TIMEn = pandas.to_datetime(turnstile_weather.TIMEn)

	dateGroup = turnstile_weather[['DATEn', 'ENTRIESn_hourly', 'rain']].groupby('DATEn', as_index=False).sum()
	dateGroup['dayOfWeek'] = pandas.DatetimeIndex(dateGroup.DATEn).weekday
	dateGroup['color'] = dateGroup.apply(getDataPointColor, axis=1)

	print dateGroup

	print (ggplot(dateGroup, aes(x = dateGroup.dayOfWeek, y = dateGroup.ENTRIESn_hourly)) +
		geom_point(aes(color = dateGroup.color)) + xlab('Day of week') + ylab('Entries') + ggtitle("Entries per day of week"))

def normalize_features(df):
	'''
	Normalize the features in the data set.
	'''
	mu = df.mean()
	sigma = df.std()
    
	if (sigma == 0).any():
		raise Exception("One or more features had the same value for all samples, and thus could " + \
						"not be normalized. Please do not include features with only a single value " + \
						"in your model.")
	df_normalized = (df - df.mean()) / df.std()
	#df_normalized = df # Use to disable the normalization.

	#print df
	#print "\n"
	#print df_normalized

	return df_normalized, mu, sigma

def compute_cost(features, values, theta):
	'''
	Compute the cost function given a set of features / values, 
	and the values for our thetas.

	This can be the same code as the compute_cost function in the lesson #3 exercises,
	but feel free to implement your own.
	'''
    
	# your code here
	m = len(values)
	sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
	cost = sum_of_square_errors / (2*m)

	#print "\nCost computed: " + str(cost)

	# Data point index for viewing progress on one particular point
	index = 170 # Good value
	#index = 270 # Bad value
	#index = 70 # Decent value
	#index = 370 # Bad value
	#index = 470 # Little less than decent

	'''
	print "Features (" + str(len(features[index])) + ")"
	print features[index]
	
	print "\nTheta (" + str(len(theta)) + ")"
	print theta
	'''
	#print "\nValue: " + str(values[index])
	#print "\nDot product: " + str(np.dot(features[index], theta))

	#print "Prediction error (index " + str(index) + ": " + str(np.dot(features[index], theta) - values[index])
	
	return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
	'''
	Perform gradient descent given a data set with an arbitrary number of features.

	This can be the same gradient descent code as in the lesson #3 exercises,
	but feel free to implement your own.
	'''

	m = len(values) * 1.0
	cost_history = []

	for i in range(num_iterations):
		cost_history.append(compute_cost(features, values, theta))
		theta = theta + alpha * (1/m) * np.dot((values - np.dot(features,theta)),features)

	return theta, pandas.Series(cost_history)

def plot_cost_history(alpha, cost_history):
	'''
	This function is for viewing the plot of your cost history.
	'''

	cost_df = pandas.DataFrame({
		'Cost_History': cost_history,
		'Iteration': range(len(cost_history))
	})

	return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
		geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

def compute_r_squared(data, predictions):

	data_average = np.mean(data)

	r_squared = 1.0 - (np.sum((data - predictions) ** 2) / (np.sum((data - data_average) ** 2)))
    
	return r_squared

def executeLinearRegression():

	turnstile_weather = pandas.read_csv('turnstile_data_master_with_weather.csv')
	#turnstile_weather = pandas.read_csv('turnstile_weather_v2.csv')
	
	turnstile_weather.DATEn = pandas.to_datetime(turnstile_weather.DATEn)
	turnstile_weather.TIMEn = pandas.to_datetime(turnstile_weather.TIMEn)
	turnstile_weather['dayOfWeek'] = pandas.DatetimeIndex(turnstile_weather.DATEn).weekday # Create data point for day of week.

	# Select Features
	# Calculated with DoW and Unit dummy values
	features = turnstile_weather[['rain', 'precipi', 'Hour', 'meantempi', 'fog', 'meandewpti', 'meanpressurei']] # R^2 = 0.469591287708
	
	# Add DoW to features using dummy variables. Gives the data point a value of 1 in the column of it's dow
	dummy_units = pandas.get_dummies(turnstile_weather['dayOfWeek'], prefix='DoW')
	features = features.join(dummy_units)
	#dummy_units = pandas.get_dummies(turnstile_weather['rain'], prefix='Rain')
	#features = features.join(dummy_units)
	# Add UNIT to features using dummy variables
	dummy_units = pandas.get_dummies(turnstile_weather['UNIT'], prefix='unit')
	features = features.join(dummy_units)

	# Values
	values = turnstile_weather['ENTRIESn_hourly']
	m = len(values)

	features, mu, sigma = normalize_features(features)
	features['ones'] = np.ones(m) # Add a column of 1s (y intercept)

	print "Features (" + str(len(features)) + " data points)"
	#print features.columns.values
	print features.describe()

	# Convert features and values to numpy arrays
	features_array = np.array(features)
	values_array = np.array(values)

	# Set values for alpha, number of iterations.
	alpha = 0.2 # please feel free to change this value
	num_iterations = 15 # please feel free to change this value

	# Initialize theta, perform gradient descent
	theta_gradient_descent = np.zeros(len(features.columns))
	theta_gradient_descent, cost_history = gradient_descent(features_array, 
															values_array, 
															theta_gradient_descent, 
															alpha, 
															num_iterations)

	print "\nTheta (" + str(len(theta_gradient_descent)) + ")"
	print theta_gradient_descent

	plot = None
	plot = plot_cost_history(alpha, cost_history)

	predictions = np.dot(features_array, theta_gradient_descent)

	print "\n"

	print plot

	#print "predictions (" + str(len(predictions)) + ")-\n"
	#print predictions
	print str(num_iterations) + " iterations, Alpha value - " + str(alpha)
	print "R^2: " + str(compute_r_squared(values_array, predictions)) + "\n"

def main():

	os.system('cls')

	#processData()
	#printSubwayWeatherStatistics()
	#printDistributionHistogram()
	#printTurnstileVisualization()
	#printDateVisualization()
	executeLinearRegression()


if __name__ == '__main__':
	main()