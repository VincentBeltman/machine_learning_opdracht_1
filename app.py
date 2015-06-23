

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
			users[data[0]] = {
				'age': int(data[1]),
				'gender': data[2],
				'occupation': data[3],
				'zipcode': data[4].strip('\n')
			}
	return users

def readMovies(genres):
	movies = {}
	with open('data/u.item') as f:
		for line in f.readlines():
			data = line.split('|')
			movies[data[0]] = {
				'title': data[1],
				# 'release_date': data[2],
				# 'video_release_date': data[3],
				# 'imdb_url': data[4]
			}
			movies[data[0]]['genres'] = []
			for index, isGenre in enumerate(data[5:]):
				if isGenre.strip('\n') == '1':
					movies[data[0]]['genres'].append(index)
	return movies

def printStatistics(genres, users, movies):
	genderStatistic = {}
	with open('data/u.data') as f:
		for index, line in enumerate(f.readlines()):
			data = line.split('\t')
			user = users[data[0]]
			movie = movies[data[1]]
			rating = int(data[2])

			for genre in movie['genres']:
				genreKey = user['gender'] + str(genre)
				if genreKey in genderStatistic:
					genderStatistic[genreKey] += 1
				else:
					genderStatistic[genreKey] = 1
	print getTop5(genderStatistic)

def getTop5(statistic):
	highestMen = [0, 0, 0, 0, 0]
	highestFemale = [0, 0, 0, 0, 0]
	resultMen = []
	resultFemale = []
	for key, val in statistic.items():
		if key.startswith('M'):
			highest = highestMen
			result = resultMen
		else:
			highest = highestFemale
			result = resultFemale
		if val > highest[-1]:
			for index, high in enumerate(highest):
				if val > high:
					highest = highest[:index] + [val] + highest[index:-1]
					result = result[:index] + [{key: val}] + result[index:-1]
					break
		if key.startswith('M'):
			highestMen = highest
			resultMen = result
		else:
			highestFemale = highest
			resultFemale = result
	return highestFemale

if __name__ == "__main__":
	genres = readGenres()
	users = readUsers()
	movies = readMovies(genres)
	data = printStatistics(genres, users, movies)
