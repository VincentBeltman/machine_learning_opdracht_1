from sklearn import cluster
import numpy as np
import math

def readGenres():
	genres = []
	with open('data/u.genre') as f:
		for line in f.readlines():
			genre = line.split('|')[0]
			if '\n' not in genre:
				genres.append(genre)
	return genres

def readUsers():
	users = {}
	with open('data/u.user') as f:
		for line in f.readlines():
			data = line.split('|')
			age = int(data[1])
			age -= age % 10
			users[data[0]] = {
				'age': '{0}-{1}'.format(age, age + 10),
				'gender': data[2],
				'occupation': data[3],
				'zipcodefirst': data[4][0]
			}
	return users

def readMovies():
	movies = {}
	with open('data/u.item') as f:
		for line in f.readlines():
			data = line.split('|')
			movies[data[0]] = {
				'title': data[1]
			}
			movies[data[0]]['genres'] = []
			for index, isGenre in enumerate(data[5:]):
				if isGenre.strip('\n') == '1':
					movies[data[0]]['genres'].append(index)
	return movies

def getUserMovieList(movies):
	userMovieList = {}
	with open('data/u.data') as f:
		for line in f.readlines():
			data = line.split('\t')
			for genre in movies[data[1]]['genres']:
				userMovieList = addOrCreate(data[0], userMovieList, [genre])
	return userMovieList

def calcGenres(userMovieList):
	otherUsers = []
	mainUser = []
	for user_id, genres in userMovieList.items():
		length = len(genres)
		old = -1
		count = 0
		result = [user_id] + [0] * 19
		for genre in sorted(genres):
			if old != -1 and genre != old:
				result[int(old) + 1] = 100 * float(count) / length
				count = 0
			count += 1
			old = genre
		if count > 0:
			result[int(old) + 1] = 100 * float(count) / length
		if user_id == '344':
			mainUser.append(result)
		else:
			otherUsers.append(result)
	return np.array(mainUser), np.array(otherUsers + mainUser)

def dbscan(userGenresCalced, mainUser):
	dbscan = cluster.DBSCAN(eps=35, min_samples=8, metric=metrics)
	#labels = dbscan.fit_predict(userGenresCalced, mainUser)
	labels = dbscan.fit(userGenresCalced).labels_
	print len(set(labels)) - (1 if -1 in labels else 0)

	# for label in labels:
	# 	print label
	# print len(labels)

def metrics(A, B):
	count = 0
	for index, item in enumerate(A[1:]):
		difference = float(item) - float(B[index])
		count += difference * difference
	return math.sqrt(count)


def addOrCreate(check, itemList, newVal):
	if check in itemList: 	itemList[check] += newVal
	else: 					itemList[check] = newVal
	return itemList

def printIt(printable): print printable

printMethod = None
try:
	from pprint import pprint
	printMethod = pprint
except Exception, e:
	printMethod = printIt

if __name__ == "__main__":
	genres = readGenres()
	movies = readMovies()
	users = readUsers()
	userMovieList = getUserMovieList(movies)
	mainUser, userGenresCalced = calcGenres(userMovieList)
	alsl = 0
	for item in mainUser:
		alsl += float(item[2])
	dbscan(userGenresCalced, mainUser)
