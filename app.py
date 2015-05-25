from numpy import *
import stringVals as sv
from sklearn import svm, tree, neighbors
from sklearn import metrics as met
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
		returnMat = zeros((numberOfLines, 14), dtype=float)
		classLabelVector = []
		colors = []
		for i, line in enumerate(filtered):
			listFromLine = []
			for j, item in enumerate(line.split(', ')):
				if j in [0,2,4,10,11,12]:
					listFromLine.append(int(item))
				elif item == '?':
					listFromLine.append(0)
				elif j == 1:
					listFromLine.append(sv.workclass.index(item))
				elif j == 3:
					listFromLine.append(sv.education.index(item))
				elif j == 5:
					listFromLine.append(sv.married.index(item))
				elif j == 6:
					listFromLine.append(sv.occupation.index(item))
				elif j == 7:
					listFromLine.append(sv.relationship.index(item))
				elif j == 8:
					listFromLine.append(sv.race.index(item))
				elif j == 9:
					listFromLine.append(sv.sex.index(item))
				elif j == 13:
					listFromLine.append(sv.country.index(item))
				elif j == 14:
					listFromLine.append(item.replace(".", ""))
			returnMat[i,:] = listFromLine[0:14]
			classLabelVector.append(listFromLine[-1])
			if listFromLine[-1] == '<=50K\n':
				colors.append(0)
			else:
				colors.append(100)

		return returnMat, classLabelVector, colors

def showScatterPlot(data, colors, idx1, idx2):
	import matplotlib.pyplot as plt
	N = 50
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

def execute(clf, dataset, trainingset, dataLabels, trainingLabels):
	# Fit
	clf = clf.fit(dataset, dataLabels)

	# Test
	testClassifier(trainingset, trainingLabels, clf)

def knn(trainingset, dataset, dataLabels, trainingLabels):
	print 'knn'
	# Gen KNN
	clf = neighbors.KNeighborsClassifier(n_neighbors=30)

	# Normalize
	dataset, ranges, mins, maxs = normalize(dataset)
	trainingset = normalize(trainingset, ranges, mins, maxs)

	# Fit and test
	execute(clf, dataset, trainingset, dataLabels, trainingLabels)

def tree(trainingset, dataset, dataLabels, trainingLabels):
	print 'tree'
	pass

def svm(trainingset, dataset, dataLabels, trainingLabels):
	print 'svm'
	pass

def naive(trainingset, dataset, dataLabels, trainingLabels):
	print 'naive'
	pass

def asd(trainingset, dataset, dataLabels, trainingLabels): # MIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKE
	print 'knn'
	pass

if __name__ == "__main__":
	algoritmes = {
		'knn': knn,
		'tree': tree,
		'svm': svm,
		'naive': naive,
		'': knn}  # MIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKEMIKE
	dataset, dataLabels, colors = readTrainingSet("adult.data")
	trainingset, trainingLabels, colors = readTrainingSet("adult.test")

	# TEST TEST TEST TEST TEST TEST TEST TEST TEST
	algoritmes[sys.argv[1]](trainingset, dataset, dataLabels, trainingLabels)

	# '#' weghalen om alles te draaien. TEST ook weghalen
	# for name, algo in algoritmes:
	#	algo()	
	
	# showScatterPlot(returnMat, colors, 10, 11)