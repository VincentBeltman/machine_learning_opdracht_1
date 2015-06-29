from sklearn import cluster
import numpy as np
import math
import app

def getUserMovieGenreList(movies):
	userMovieGenreList = {}
	with open('data/u.data') as f:
		for line in f.readlines():
			data = line.split('\t')
			for genre in movies[data[1]]['genres']:
				userMovieGenreList = app.addOrCreate(data[0], userMovieGenreList, [genre])
	return userMovieGenreList

def getUserMovieList(movies, users):
	userMovieList = np.zeros((len(users.keys()), len(movies.keys())), dtype=float)
	with open('data/u.data') as f:
		for line in f.readlines():
			data = line.split('\t')
			userMovieList[int(data[0])-1,][int(data[1])-1,] = data[2]
	return userMovieList

def calcGenres(userMovieList, size_test=0):
	size_test *= -1
	users = []
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
		users.append(result)
	return np.array(users[size_test:]), np.array(users[:size_test])

def dbscan(userGenresCalced):
	dbscan = cluster.DBSCAN(eps=40, min_samples=8, metric=metrics)
	labels = dbscan.fit(userGenresCalced).labels_
	print len(set(labels)) - (1 if -1 in labels else 0)

def kmeans(userGenresCalced, mainUser, n_clusters):
	kmeans = cluster.KMeans(n_clusters=n_clusters)
	kmeans.fit(userGenresCalced)
	print kmeans.predict(mainUser)

def metrics(A, B):
	count = 0
	for index, item in enumerate(A[1:]):
		difference = float(item) - float(B[index])
		count += difference * difference
	return math.sqrt(count)

def pearsonDull(x, y):
	n = len(x)
	sum_x = float(sum(x))
	sum_y = float(sum(y))
	sum_x_sq = sum(map(lambda x: pow(x, 2), x))
	sum_y_sq = sum(map(lambda x: pow(x, 2), y))
	psum = sum([x[i] * y[i] for i in x])
	num = psum - (sum_x * sum_y/n)
	den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
	if den == 0: return 0
	return num / den

pearson = None
try:
	from scipy.stats.stats import pearsonr
	pearson = pearsonr
except Exception, e:
	pearson = pearsonDull

def clustering(movies):
	userMovieList = getUserMovieGenreList(movies)
	testUsers, userGenresCalced = calcGenres(userMovieList, size_test=20)
	clusters = dbscan(userGenresCalced)
	for i in range(5):
		app.P(kmeans(userGenresCalced, testUsers, 18))

def recommend(movies, users, genres, userToRecommend):
	print "Recommendations van user:", userToRecommend
	userMovieList = getUserMovieList(movies, users)
	nearest = computeNearestNeighbor(userToRecommend, userMovieList)
	app.P(users[str(userToRecommend)])
	recommendations = [{}, {}, {}, {}, {}]
	print '\n\nTop 5 gelijke gebruikers'
	recommendedUsersMovies = [movie for movie, rating in enumerate(userMovieList[userToRecommend]) if rating > 3]
	for top, item in enumerate(nearest):
		top += 1
		print '\nNummer', top
		print '\tPearson correlation:', item[0][0]
		print '\tUser:'
		app.P(users[str(item[1])], indent=4)
		for movie, rating in enumerate(userMovieList[item[1]]):
			if rating > 0:
				recommendations[0] = app.addOrCreate(movie, recommendations[0], 1)
				recommendations[3] = app.addOrCreate(movie, recommendations[3], 1 * float(top) / 10)
				if movie not in recommendedUsersMovies:
					recommendations[1] = app.addOrCreate(movie, recommendations[1], 1)
				if rating >= 3:
					recommendations[2] = app.addOrCreate(movie, recommendations[2], 1)
				if movie not in recommendedUsersMovies and rating >= 3:
					recommendations[4] = app.addOrCreate(movie, recommendations[4], 1 * float(top) / 10)
	print '\n\nTop 5 movies aan de hand van andere gebruikers'
	for recommendationNr, recommendedMovies in enumerate(recommendations):
		top = 1
		print "\nAlgoritme", recommendationNr + 1
		for movie_count in app.getTopN(recommendedMovies):
			# Make a clown
			movie = dict(movies[str(movie_count.keys()[0])])
			print '\nNummer', top
			print '\tMovie:', movie['title']
			print '\tGenres', [genres[genre] for genre in movie['genres']]
			top += 1

def computeNearestNeighbor(recommendUserId, users):
	distances = []
	for user_id, movies in enumerate(users):
		if user_id != recommendUserId:  # Niet zichzelf
			distance = pearson(users[user_id], users[recommendUserId])
			distances.append((distance, user_id))
	distances.sort()
	return distances[:-6:-1]

if __name__ == "__main__":
	genres, movies, users = app.readGenres(), app.readMovies(), app.readUsers()
	clustering(movies)
	recommend(movies, users, genres, userToRecommend=407)
