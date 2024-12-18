from flask import Flask, render_template, request, jsonify
from surprise import dump
import pandas as pd
from surprise import SVD, Dataset, Reader

app = Flask(__name__)

# Load the pre-trained model
_, svd = dump.load('./svd_model.pkl')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    genre = request.json['genre']
    recommendations = recommend_movies(int(user_id), genre)
    return jsonify(recommendations)

def recommend_movies(user_id, genre, top_n=5):
    genre_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    predictions = []
    for movie_id in genre_movies['movieId'].unique():
        predicted_rating = svd.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))

    # Sort the predictions based on estimated ratings and get the top n
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]
    print(top_predictions)
    # Return the top n movies' titles and their predicted ratings
    top_movies = [(movies.loc[movies['movieId'] == pred[0], 'title'].iloc[0], pred[1]) for pred in top_predictions]
    return top_movies

if __name__ == '__main__':
    app.run(debug=True)