import pandas as pd
from surprise import SVD, Dataset, Reader, dump

# Load data and train SVD model
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
data = pd.merge(ratings, movies, on='movieId')
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

# Save the model
dump.dump('./svd_model.pkl', algo=svd)