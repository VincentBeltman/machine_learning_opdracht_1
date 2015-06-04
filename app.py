from numpy import *
from stringVals import sv
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation
from sklearn.metrics import *

def readTrainingSet(name, N=25):
	with open(name) as dataset:
		lines = dataset.readlines()
		filtered = []
		numberOfLines = 0
		for line in lines:
			line = line.split(',')
			if len(line) == 26:
				for i, item in enumerate(line):
					if item == "?":
						break
				else:
					filtered.append(line)
					numberOfLines += 1
		returnMat = zeros((numberOfLines, N), dtype=float)
		answers = []

		maxAnswer = 0
		minAnswer = 45400

		for i, line in enumerate(filtered):
			listFromLine = []
			add = True
			for j, item in enumerate(line):
				if j in [2, 3, 4, 5, 6, 7, 8, 14, 15, 17]:
					listFromLine.append(float(sv[j].index(item)))
				elif j in [0, 1, 9, 10, 11, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24]:
					listFromLine.append(float(item))
				elif j == 25:
					item = float(item)
					answers.append(item)
					minAnswer = item if item < minAnswer else minAnswer
					maxAnswer = item if item > maxAnswer else maxAnswer
			else:
				returnMat[i,:] = listFromLine
		gem = (maxAnswer + minAnswer) / 2
		colors = []
		for answer in answers:
			if answer > gem:
				colors.append('r')
			else:
				colors.append('g')

		return returnMat, answers, colors

def scatter(xas, yas):
	plt.legend(loc=2)
	plt.xlabel("ages")
	plt.ylabel("net worths")
	plt.scatter(xas, yas, c='g')

def plot(xas, line):
	plt.plot(xas, line, color='red', linewidth=3)
	plt.show()

if __name__ == "__main__":
	trainingset, trainingLabels, colors = readTrainingSet("autoprice.txt")
	clf = linear_model.LinearRegression()
	kf = cross_validation.KFold(len(trainingset), n_folds=8, shuffle=True)

	for train_index, test_index in kf:
		clf.fit(trainingset[train_index], array(trainingLabels)[train_index])
		print clf.score(trainingset[test_index], array(trainingLabels)[test_index])
		print ""
		# scatter(trainingset[train_index][:,N], array(trainingLabels)[train_index])
		# plot(trainingset[test_index][:,N], clf.predict(trainingset[test_index]))
