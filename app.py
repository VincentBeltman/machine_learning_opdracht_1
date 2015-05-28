from numpy import *
import stringVals as sv

def readTrainingSet(name):
	with open(name) as dataset:
		lines = dataset.readlines()
		filtered = []
		numberOfLines = 0
		for line in lines:
			if len(line.split(', ')) == 15:
				filtered.append(line)
				numberOfLines += 1
		returnMat = zeros((numberOfLines, 11), dtype=float)
		classLabelVector = []
		colors = []
		for i, line in enumerate(filtered):
			listFromLine = []
			for j, item in enumerate(line.split(', ')):
				if j in [0,12]:
					listFromLine.append(int(item))
				elif j == 10:
					if not(int(item) == 0):
						listFromLine.append(int(item))
				elif j == 11:
					if not(int(item) == 0) or len(listFromLine) == 8:
						listFromLine.append(int(item) * -1)
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
					label = item.replace(".", "")
					classLabelVector.append(label)
					if label == '<=50K\n':
						colors.append('g')
					else:
						colors.append('r')

			returnMat[i,:] = listFromLine

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
	#Dictonary with params
	params = [{"n_neighbors":10, "algorithm":"auto"}, {"n_neighbors":20, "algorithm":"auto"}, {"n_neighbors":30, "algorithm":"auto"}, {"n_neighbors":20, "algorithm":"kd_tree"}, {"n_neighbors":20, "algorithm":"ball_tree"}]
	#Normalize
	dataset, ranges, mins, maxs = normalize(dataset)
	trainingset = normalize(trainingset, ranges, mins, maxs)
	print "default params"
	clf = KNeighborsClassifier()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	
	for paramDict in params:
		print("Params: n_neighbors = "+str(paramDict["n_neighbors"]) + " algorithm = " + str(paramDict["algorithm"]));
		clf = KNeighborsClassifier(n_neighbors = paramDict["n_neighbors"] , algorithm = paramDict["algorithm"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)

def tree(trainingset, dataset, trainingLabels, dataLabels):
	#Dictonary with params
	params = [{"max_features":3 , "max_depth": 10},{"max_features":5 , "max_depth": 15},{"max_features":1 , "max_depth": 20},{"max_features":10 , "max_depth": 2},{"max_features":5 , "max_depth": 10}]
	#Normalize
	dataset, ranges, mins, maxs = normalize(dataset)
	trainingset = normalize(trainingset, ranges, mins, maxs)
	print "default params"
	clf = DecisionTreeClassifier()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	
	for paramDict in params:
		print("Params: max_features = "+str(paramDict["max_features"]) + " max_depth = " + str(paramDict["max_depth"]));
		clf = DecisionTreeClassifier(max_features = paramDict["max_features"] , max_depth = paramDict["max_depth"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)

def svm(trainingset, dataset, trainingLabels, dataLabels):
	#Dictonary with params
	params = [{"c":10.0 , "gamma": 5.0},{"c":random.uniform(0.5, 8.5) , "gamma":random.uniform(0.5, 8.5)}  ,{"c":5.0 , "gamma":10.0} ,{"c":3.0 , "gamma":5.0} , {"c":2.0 , "gamma":1.5} , {"c":2.0 , "gamma":2.0} , {"c":2.0 , "gamma":3.0}]
	#Normalize
	dataset, ranges, mins, maxs = normalize(dataset)
	trainingset = normalize(trainingset, ranges, mins, maxs)
	print "default params"
	clf = SVC()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)
	
	for paramDict in params:
		print("Params: c = "+str(paramDict["c"]) + " gamma = " + str(paramDict["gamma"]));
		clf = SVC(C = paramDict["c"] , gamma = paramDict["gamma"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)

def naive(trainingset, dataset, trainingLabels, dataLabels):
	clf = GaussianNB()
	print "default params"
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)

def randomTree(trainingset, dataset, trainingLabels, dataLabels): 
	params = [{"n_estimators":5 , "max_features":5},{"n_estimators":15 , "max_features":8} , {"n_estimators":8 , "max_features":11}, {"n_estimators":20 , "max_features":9} , {"n_estimators":20 , "max_features":7}]
	print "default params"
	clf = RandomForestClassifier()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)

	for paramDict in params:
		print("Params: n_estimators = "+str(paramDict["n_estimators"]) + " max_features = " + str(paramDict["max_features"]));
		clf = RandomForestClassifier(n_estimators = paramDict["n_estimators"] , max_features = paramDict["max_features"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)

def adaBoost(trainingset, dataset , trainingLabels , dataLabels):
	params = [{"n_estimators":5} ,{"n_estimators":15} , {"n_estimators":20} , {"n_estimators":25} , {"n_estimators":35}]
	dataset, ranges, mins, maxs = normalize(dataset)
	trainingset = normalize(trainingset, ranges, mins, maxs)
	print "default params"
	clf = AdaBoostClassifier()
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)

	for paramDict in params:
		print("Params: n_estimators = "+str(paramDict["n_estimators"]));
		clf = AdaBoostClassifier(n_estimators = paramDict["n_estimators"])
		execute(clf, dataset, trainingset, dataLabels, trainingLabels)

if __name__ == "__main__":
	algoritmes = {
		'knn': knn,
		'tree': tree,
		'svm': svm,
		'naive': naive,
		'randomTree': randomTree, 
		'adaBoost': adaBoost} 
	trainingset, trainingLabels, colors = readTrainingSet("adult.data")
	dataset, dataLabels, colors = readTrainingSet("adult.test")

	# matplotlib
	# for i in range(0,11):
	# 	for j in range(0,11):
	# 		showScatterPlot(trainingset, colors, i, j)

	for name, algo in algoritmes.items():
		print name
		algo(trainingset, dataset, trainingLabels, dataLabels)	
	
