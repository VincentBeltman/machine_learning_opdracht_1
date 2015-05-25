from numpy import *
import stringVals as sv
from sklearn import svm, tree, neighbors
from sklearn import metrics as met
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys


def readTrainingSet(name):
	with open(name) as dataset:
		lines = dataset.readlines()
		filtered = []
		numberOfLines = 0
		for line in lines:
			if len(line.split(', ')) == 15:
				filtered.append(line)
				numberOfLines += 1
		returnMat = zeros((numberOfLines, 13), dtype=float)
		classLabelVector = []
		colors = []
		for i, line in enumerate(filtered):
			listFromLine = []
			for j, item in enumerate(line.split(', ')):
				if j in [0,2,4,10,11,12]:
					# Is het capital loss of gain? Voeg alleen toe als de waarde niet nul is
					if (j in [10, 11] and not(item == '0')) or True:  
						listFromLine.append(int(item))
				elif item == '?':
					listFromLine.append(0)
				elif j == 1:
					listFromLine.append(sv.workclass.index(item) +0)
				elif j == 3:
					listFromLine.append(sv.education.index(item) +0)
				elif j == 5:
					listFromLine.append(sv.married.index(item) +0)
				elif j == 6:
					listFromLine.append(sv.occupation.index(item)+0)
				elif j == 7:
					listFromLine.append(sv.relationship.index(item)+0)
				elif j == 8:
					listFromLine.append(sv.race.index(item)+0)
				elif j == 9:
					listFromLine.append(sv.sex.index(item)+0)
				elif j == 13:
					listFromLine.append(sv.country.index(item)+0)
				elif j == 14:
					listFromLine.append(item.replace(".", ""))
			returnMat[i,:] = listFromLine[0:13]
			classLabelVector.append(listFromLine[-1])
			if listFromLine[-1] == '<=50K\n':
				colors.append('g')
			else:
				colors.append('r')

		return returnMat, classLabelVector, colors

def showScatterPlot(data, colors, idx1, idx2):
	import matplotlib.pyplot as plt
	plt.scatter(data[:,idx1], data[:,idx2], c=colors)
	plt.show()

def normalize(dataset, ranges=None, mins=None, maxs=None):
	returnParams = ranges is None
	if ranges is None:
		mins = dataset.min(0)
		maxs = dataset.max(0)
		ranges = maxs - mins
	normalized = zeros(shape(dataset))
	rowCount = dataset.shape[0]
	normalized = dataset - tile(mins, (rowCount, 1))
	normalized = normalized / tile(ranges, (rowCount, 1))
	if returnParams:
		return normalized, ranges, mins, maxs
	else:
		return normalized

def testClassifier(dataset, labels, clf):
	errorCount = 0.0

	classifierResult = clf.predict(dataset)
	i = 0
	for result in classifierResult:
		if (result != labels[i]):
			errorCount += 1.0
		i += 1

	print "Error rate: {0}".format(errorCount / float(dataset.shape[0]))

def execute(clf, trainingset, dataset, trainingLabels, dataLabels):
	# Fit
	clf = clf.fit(trainingset, trainingLabels)

	# Test
	testClassifier(dataset, dataLabels, clf)

def knn(trainingset, dataset, trainingLabels, dataLabels):
	print 'knn'
	# Gen KNN
	clf = neighbors.KNeighborsClassifier(n_neighbors=30)

	# Normalize
	trainingset, ranges, mins, maxs = normalize(trainingset)
	dataset = normalize(dataset, ranges, mins, maxs)

	# Fit and test
	execute(clf, trainingset, dataset, trainingLabels, dataLabels)

def tree(trainingset, dataset, trainingLabels, dataLabels):
	print 'tree'
	pass

def svm(trainingset, dataset, trainingLabels, dataLabels):
	print 'svm'
	#Dictonary with params
	params = [{"c":10.0 , "gamma": 5.0},{"c":random.uniform(0.5, 8.5) , "gamma":random.uniform(0.5, 8.5)}  ,{"c":5.0 , "gamma":10.0} ,{"c":3.0 , "gamma":5.0} , {"c":2.0 , "gamma":1.5} , {"c":2.0 , "gamma":2.0} , {"c":2.0 , "gamma":3.0}]
	#Normalize
	dataset, ranges, mins, maxs = normalize(dataset)
	trainingset = normalize(trainingset, ranges, mins, maxs)
	print "default params"
	clf = SVC()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	
	for paramDict in params:
		print("Params c= "+str(paramDict["c"]) + "gamma =" + str(paramDict["gamma"]));
		clf = SVC(C = paramDict["c"] , gamma = paramDict["gamma"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)

	pass

def naive(trainingset, dataset, trainingLabels, dataLabels):
	print 'naive'
	clf = GaussianNB()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	pass

def randomTree(trainingset, dataset, trainingLabels, dataLabels): 
	print 'randomTree'
	params = [{"n_estimators":5 , "max_features":5},{"n_estimators":15 , "max_features":8} , {"n_estimators":8 , "max_features":11}, {"n_estimators":20 , "max_features":9} , {"n_estimators":20 , "max_features":7}]
	print "default params"
	clf = RandomForestClassifier()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)

	for paramDict in params:
		print("Params n_estimators= "+str(paramDict["n_estimators"]) + "max_features =" + str(paramDict["max_features"]));
		clf = RandomForestClassifier(n_estimators = paramDict["n_estimators"] , max_features = paramDict["max_features"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	
	pass

def adaBoost(trainingset, dataset , trainingLabels , dataLabels):
	print 'adaBoost'
	params = [{"n_estimators":5} ,{"n_estimators":15} , {"n_estimators":20} , {"n_estimators":25} , {"n_estimators":35}]
	print "default params"
	clf = AdaBoostClassifier()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)

	for paramDict in params:
		print("Params n_estimators= "+str(paramDict["n_estimators"]));
		clf = AdaBoostClassifier(n_estimators = paramDict["n_estimators"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	pass

if __name__ == "__main__":
	algoritmes = {
		'knn': knn,
		'tree': tree,
		'svm': svm,
		'naive': naive,
		'randomTree': randomTree , 
		'adaBoost': adaBoost} 
	trainingset, trainingLabels, colors = readTrainingSet("adult.data")
	showScatterPlot(trainingset, colors, 0, 1)
	#dataset, dataLabels, colors = readTrainingSet("adult.test")

	# TEST TEST TEST TEST TEST TEST TEST TEST TEST
	#algoritmes[sys.argv[1]](trainingset, dataset, trainingLabels, dataLabels)

	# '#' weghalen om alles te draaien. TEST ook weghalen
	# for name, algo in algoritmes:
	#	algo(trainingset, dataset, dataLabels, trainingLabels)	
	
