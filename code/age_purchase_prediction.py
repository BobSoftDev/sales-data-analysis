# Term project written for CPSC 340 
# Rui Cao
# DEC. 21, 2018

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle 
import argparse
import utils


import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import accuracy_score

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#from sklearn import linear_model, datasets
def main():

	# functions to clean the data
	def change_year(year):
		if year == '4+':
			return 4
		else:
			return int(year)
	def change_gender(gender):
		if gender == 'F':
			return 0
		else:
			return 1
	def change_age(age):
		if age == '0-17':
			return 0
		elif age == '18-25':
			return 1
		elif age == '26-35':
			return 2
		elif age == '36-45':
			return 3
		elif age == '46-50':
			return 4
		elif age == '51-55':
			return 5
		else:
			return 6
	def change_city(city_category):
		if city_category == 'A':
			return 0
		elif city_category == 'B':
			return 1
		else:
			return 2
	
	def cal_error(value, predict):

		error = np.mean(np.sum((value - predict))/value)
		return error

	X = pd.read_csv('BlackFriday.csv')
	N, d = X.shape
	print(N,d)
	# fill missing values with 0
	# (?) need to calculate percentage of missing value?
	X = X.fillna(0)
	# change gender to 0 and 1
	X['Gender'] = X['Gender'].apply(change_gender)
	# change age to 0 to 6
	X['Age'] = X['Age'].apply(change_age)
	# change city categories to 0 to 2
	X['City_Category'] = X['City_Category'].apply(change_city)
	# change the year to integer
	X['Stay_In_Current_City_Years'] = X['Stay_In_Current_City_Years'].apply(change_year)
	# Make y matrix to be the age
	y = np.zeros((N,1))
	y = X.values[:,3]
	y = y.astype('int')

	# X_no_age matrix deletes the Age column in the original dataset 
	X_no_age = X
	X_no_age = X_no_age.drop(columns = ['User_ID','Product_ID','Age'])

	# split the data into training and test set using sklearn build-in function
	# the test_size = 0.2
	# number of test examples = 107516
	# number of training examples = 430061
	X_train, X_test, y_train, y_test = train_test_split(X_no_age, y, test_size=0.2)

	# X_no_purchase matrix deletes the Purchase column in the original dataset
	X_no_purchase = X
	X_no_purchase = X_no_purchase.drop(['User_ID', 'Product_ID','Purchase'], axis = 1)
	print(X_no_purchase.shape)

	# delete ids: 4623.626428752559 4633.763914844436
	# delete id and product category: 4959.802246693021 4958.728468583989
	# delete id and age: 4637.530137767859 4617.998339275238
	yp = np.zeros((N,1))
	yp = X.values[:,11]
	yp = yp.astype('int')

	Xptrain, Xptest, yptrain, yptest = train_test_split(X_no_purchase, yp, test_size=0.2)



	if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('-q','--question', required=True)

		io_args = parser.parse_args()
		question = io_args.question

		if question == 'knn':
			depth = 5
			model = KNeighborsClassifier(n_neighbors=depth)
			model.fit(X_train, y_train)

			y_pred = model.predict(X_train)
			tr_error = np.mean(y_pred != y_train)

			y_pred = model.predict(X_test)
			te_error = np.mean(y_pred != y_test)
			print("KNN with depth %f" % depth)
			print("Training error: %.3f" % tr_error)
			print("Validation error: %.3f" % te_error)

			# result: 
			# k=10: Training error: 0.526 Testing error: 0.630
			# k=3: Training error: 0.405 Testing error: 0.669
			# k=5: Training error: 0.462 Testing error: 0.650


		elif question == 'randomforest':
			# Random forest using information gain
			# number of trees = 50
			def evaluate_model(model):


				model.fit(X_train,y_train)

				y_pred = model.predict(X_train)
				tr_error = np.mean(y_pred != y_train)

				y_pred = model.predict(X_test)
				te_error = np.mean(y_pred != y_test)
				print("	Training error: %.3f" % tr_error)
				print("	Validation error: %.3f" % te_error)

			print("Random Forest:")
			evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators = 30))

			# result:	Training error: 0.008  Testing error: 0.410 (n_estimator = 50)
			# Training error: 0.037  Testing error: 0.412 (n_estimator = 10)
			# Training error: 0.538 Validation error: 0.536 (n_estimator = 50, max_depth=5)

		elif question == "linearmodel":

		#	print(X.info(verbose=True))
			model = LinearRegression().fit(X_train, y_train)
			#utils.test_and_plot(model,Xptrain,yptrain,Xptest,yptest,title="Least Squares, with bias",filename="least_squares_with_bias.pdf")
			#print(model.coef_, model.intercept_)
			yhat_train = model.predict(X_train)
			yhat = model.predict(X_test)

			print(yhat.shape, yhat_train.shape)

			Ntest = y_test.shape[0]
			Ntrain = y_train.shape[0]

			onestest = np.ones((Ntest, 1))
			onestrain = np.ones((Ntrain, 1))

			#trainpercent = sqrt(mean_squared_error(onestrain, np.divide(yhat_train,y_train)))
			#testpercent = sqrt(mean_squared_error(onestest, np.divide(yhat,y_test)))

			testpercent = cal_error(y_train, yhat_train)
			trainpercent = cal_error(y_test, yhat)

			print("Training error percentage: %.3f" % trainpercent)
			print("Testing error percentage: %.3f" % testpercent)

			'''
			print(sqrt(mean_squared_error(yptrain, yhat_train)))
			print(sqrt(mean_squared_error(yptest, yhat)))
			#prediction = pd.DataFrame({'Actual': yptest, 'Prediction': yhat})
			#print(prediction.head)
'''
		elif question == 'poly':
			model = make_pipeline(PolynomialFeatures(2), Ridge())
			model.fit(Xptrain, yptrain)
			yhat_train = model.predict(Xptrain)
			yhat = model.predict(Xptest)

			Ntest = yptest.shape[0]
			Ntrain = yptrain.shape[0]

			onestest = np.ones((Ntest, 1))
			onestrain = np.ones((Ntrain, 1))

			trainpercent = sqrt(mean_squared_error(onestrain, np.divide(yhat_train,yptrain)))
			testpercent = sqrt(mean_squared_error(onestest, np.divide(yhat,yptest)))

			print(trainpercent, testpercent)
'''
			print("Training mean square error = %.3f" % sqrt(mean_squared_error(yptrain, yhat_train)))

			rms_test = sqrt(mean_squared_error(yptest, yhat))
			print("Testing mean square error = %.3f" % rms_test)
			'''



main()