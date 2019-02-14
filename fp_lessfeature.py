#!/usr/bin/env python
#展示这段代码是因为比较擅长python，并且可以展示自己的machine learning能力

from __future__ import division
from collections import defaultdict
from sklearn import *
import numpy as np
from time import time
import sys
import random
from math import log

# file = 'diamonds.shuffle.5k.csv'
# file = 'diamonds.csv'
file = 'diamonds.sort.csv'
data = defaultdict(lambda: defaultdict())
# num_class = 5
datasize = 53940
cutList = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
colorList = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarityList = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
cut, color, clarity = {}, {}, {}
xishu = 100000000000

def genDict():
	dictList = [cut, color, clarity]
	for i, listname in enumerate([cutList, colorList, clarityList]):
		for j, key in enumerate(listname):
			dictList[i][key] = j

def readfile(filename):
	title = defaultdict()
	statResult = defaultdict(set)
	for i, line in enumerate(open(filename)):
		col = map(lambda x: float(x) if x[0] != '"' else x[1:-1], line.strip().split(',')[1:])
		# print col
		for j in xrange(len(col)):
			if i == 0:
				title[j] = col[j]
			else:
				# if title[j] == 'price':
				# 	col[j] = log(col[j]) * xishu
				data[i][title[j]] = col[j]
				statResult[title[j]].add(col[j])
	return statResult

def feature_mapping(statResult):
	feature2index = dict()
	index2feature = dict()
	for t in statResult:
		if t == 'price':
			continue
		elif t in {'carat', 'x', 'y', 'z'}:
			feature2index[t] = len(feature2index)
			index2feature[len(index2feature)] = t
		elif t in {'cut', 'color', 'clarity'}:
			for v in statResult[t]:
				feature2index[t,v] = len(feature2index)
				index2feature[len(index2feature)] = (t,v)
			feature2index[t] = len(feature2index)
			index2feature[len(index2feature)] = t
		else:
			for v in statResult[t]:
				feature2index[t,v] = len(feature2index)
				index2feature[len(index2feature)] = (t,v)
	dimension = len(feature2index)
	print "Dimensionality: ", dimension
	return feature2index, dimension, index2feature

def map_data(feature2index, dimension):
	X, Y, Z = [], [], []
	# dimension = len(feature2index)
	for i in data:
		feat_vec = np.zeros(dimension)
		for t in data[i]:
			v = data[i][t]
			if t == 'price':
				# price_class = int(i // ((datasize + 1) / num_class))
				range_class = (max(statResult['price']) - 0) / num_class
				for k in xrange(num_class):
					if v <= min(statResult['price']) + (k+1) * range_class:
						price_class = k
						break
				Y.append(price_class)
				Z.append(int(v))
			elif t in {'carat', 'x', 'y', 'z'}:
				feat_vec[feature2index[t]] = v
			else:
				if (t, v) in feature2index:
					feat_vec[feature2index[t, v]] = 1
				if t == 'cut':
					feat_vec[feature2index[t]] = cut[v]
				elif t == 'color':
					feat_vec[feature2index[t]] = color[v]
				elif t == 'clarity':
					feat_vec[feature2index[t]] = clarity[v]
		X.append(feat_vec)
	return X,Y,Z

def train(XY,C,epoch,index2feature):
	X, Y = zip(* XY)
	classifier_models = dict()
	classifier_models['GBClassifier'] = ensemble.GradientBoostingClassifier()
	classifier_models['SVClinearClassifier'] = svm.SVC(kernel = 'linear', C=C)
	classifier_models['SVCpoly2Classifier'] = svm.SVC(kernel = 'poly', degree = 2, coef0 = 1, C=C)
	classifier_models['SVCpoly3Classifier'] = svm.SVC(kernel = 'poly', degree = 3, coef0 = 1, C=C)
	classifier_models['SVMrbfClassifier'] = svm.SVC(kernel = 'rbf', C=C)
	classifier_models['RandomForestClassifier'] = ensemble.RandomForestClassifier()
	classifier_models['AdaBoostClassifier'] = ensemble.AdaBoostClassifier()
	classifier_models['DecisionTreeClassifier'] = tree.DecisionTreeClassifier()
	classifier_models['GaussianNBClassifier'] = naive_bayes.GaussianNB()
	classifier_models['KNNClassifier'] = neighbors.KNeighborsClassifier()
	classifier_models['MLPClassifier'] = neural_network.MLPClassifier()
	classifier_models['SGDClassifier'] = linear_model.SGDClassifier(n_iter=epoch,verbose=0)
	
	# kfold = model_selection.KFold(n_splits=10, shuffle=False, random_state=None)
	for name in classifier_models:
		print '----------------', name, '----------------'
		start = time()
		scores = cross_validation.cross_val_score(classifier_models[name], X, Y, cv=5)
		end = time()
		print 'running time:', end - start
		print scores
		mean, std = '{:.2%}'.format(scores.mean()), '{:.2%}'.format(scores.std())
		print mean, std
		print '\n'


def regression(XY, epoch, index2feature, C):
	X, Y = zip(* XY)
	regressor_models = dict()
	regressor_models['LinearRegressor'] = linear_model.LinearRegression()
	regressor_models['AdaBoostRegressor'] = ensemble.AdaBoostRegressor()
	regressor_models['RandomForestRegressor'] = ensemble.RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
	regressor_models['SGDRegressor'] = linear_model.SGDRegressor(n_iter=epoch)
	# regressor_models['SVRlinearRegressor'] = svm.SVR(kernel='linear')
	# regressor_models['SVRpoly2Regressor'] = svm.SVR(kernel = 'poly', degree = 2)
	regressor_models['MLPRegressor'] = neural_network.MLPRegressor()
	regressor_models['DecisionTreeRegressor'] = tree.DecisionTreeRegressor()
	regressor_models['KNNRegressor'] = neighbors.KNeighborsRegressor()

	for name in regressor_models:
		print '----------------', name, '----------------'
		if name != 'LogisticRegressor':
			start = time()
			scores = cross_validation.cross_val_score(regressor_models[name], X, Y, cv=5)
			end = time()
			print 'running time:', end - start
			print scores
			mean, std = '{:.2%}'.format(scores.mean()), '{:.2%}'.format(scores.std())
			print 'mean:', mean, 'std:', std

		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3)
		regressor_models[name].fit(X_train, Y_train)
		Y_train_predict = regressor_models[name].predict(X_train)
		MSE_train = metrics.mean_squared_error(Y_train, Y_train_predict)
		Y_test_predict = regressor_models[name].predict(X_test)
		MSE_test = metrics.mean_squared_error(Y_test, Y_test_predict)
		print 'MSE_train:','{:.2f}'.format(MSE_train), 'MSE_test:', '{:.2f}'.format(MSE_test)
		print '\n'



if __name__ == '__main__':
	num_class = int(sys.argv[1])
	C = float(sys.argv[2])
	epoch = int(sys.argv[3])
	genDict()
	statResult = readfile(file)
	feature2index, dimension, index2feature = feature_mapping(statResult)
	X,Y,Z = map_data(feature2index, dimension)

	XY = zip(X,Y)
	random.shuffle(XY)
	train(XY,C,epoch,index2feature)

	XZ = zip(X,Z)
	random.shuffle(XZ)
	regression(XZ, epoch, index2feature, C)
