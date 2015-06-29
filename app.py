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
				'zipcodefirst': data[4][0],
				'user_id': data[0]
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

def printStatistics(genres, users, movies):
	top5Items = [('gender', {}), ('occupation', {}), ('zipcodefirst', {}), ('age', {})]
	genreRanking = {}
	totalMovieRanking = {}
	with open('data/u.data') as f:
		for index, line in enumerate(f.readlines()):
			data = line.split('\t')
			user = users[data[0]]
			movie_id = data[1]
			movie = movies[movie_id]
			rating = int(data[2])
			totalMovieRanking = addOrCreate(movie_id, totalMovieRanking, [rating])
			for genre in movie['genres']:
				genreName = genres[genre]
				for attr, statistic in top5Items:
					item = user[attr]
					if item in statistic:
						statistic[item] = addOrCreate(genreName, statistic[item], 1)
					else:
						statistic[item] = { genreName: 1 }
				if genreName in genreRanking:
					genreRanking[genreName] = addOrCreate(movie_id, genreRanking[genreName], [rating])
				else:
					genreRanking[genreName] = { movie_id: [rating] }
	movieRankingResult = {}
	for movie_id, ratings in totalMovieRanking.items():
		total = 0
		for rating in ratings:
			total += rating
		movieRankingResult[movies[movie_id]['title']] = float(total) / len(ratings)
	genreRankingResult = {}
	for genre, movieRankings in genreRanking.items():
		for movie_id, ratings in movieRankings.items():
			total = 0
			for rating in ratings:
				total += rating
			genreRankingResult = addOrCreate(genre, genreRankingResult, {movies[movie_id]['title']: float(total) / len(ratings)}, True)

	for statistic in top5Items:
		P(getTop5Ranking(statistic[1]))
	print "\nMovie per genre ranking"
	P(getTop5Ranking(genreRankingResult))
	print "\nMovie ranking top 10"
	P(getTopN(movieRankingResult, 10))

def getTop5Ranking(statistic):
	return {key: getTopN(items) for key, items in statistic.items()}

def getTopN(statistic, N=5):
	highest = [0] * N
	resultList = []
	for key, count in statistic.items():
		if count > highest[-1]:
			for index, high in enumerate(highest):
				if count > high:
					highest = highest[:index] + [count] + highest[index:-1]
					resultList = resultList[:index] + [{key: count}] + resultList[index:]
					break
	return resultList[:N]

def addOrCreate(check, itemList, newVal, isDict=False):
	if check in itemList:
		if isDict: 	itemList[check].update(newVal)
		else: 		itemList[check] += newVal
	else: 	itemList[check] = newVal
	return itemList

def printIt(printable): print printable

P = None
try:
	from pprint import pprint
	P = pprint
except Exception, e:
	P = printIt


if __name__ == "__main__":
	genres = readGenres()
	users = readUsers()
	movies = readMovies()
	data = printStatistics(genres, users, movies)
