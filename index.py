import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Read user data from users.json
with open('users.json', 'r') as f:
    users_data = json.load(f)

# Read movie data from movies.json
with open('movies.json', 'r') as f:
    movies_data = json.load(f)

# Read movie ratings data from movie_ratings.json
with open('movie_ratings.json', 'r') as f:
    movie_ratings_data = json.load(f)

# Extract user IDs, movie IDs, and ratings from movie_ratings_data
user_ids = [rating['user_id'] for rating in movie_ratings_data]
movie_ids = [rating['movie_id'] for rating in movie_ratings_data]
ratings = [rating['rating'] for rating in movie_ratings_data]

# Find the maximum user ID and movie ID
max_user_id = max(user_ids)
max_movie_id = max(movie_ids)

# Create a user-item matrix from the ratings data
user_movie_ratings = np.zeros((max_user_id, max_movie_id))

for rating in movie_ratings_data:
    user_id = rating['user_id'] - 1  # Adjust index to start from 0
    movie_id = rating['movie_id'] - 1  # Adjust index to start from 0
    user_movie_ratings[user_id][movie_id] = rating['rating']

# Calculate item-item similarity matrix using cosine similarity
item_similarity_matrix = cosine_similarity(user_movie_ratings.T)

def get_top_similar_items(movie_id, n=3):
    # Get similarity scores for the given movie_id
    movie_similarity_scores = item_similarity_matrix[movie_id - 1]

    # Sort indices of movies by their similarity scores (descending order)
    similar_movie_indices = np.argsort(movie_similarity_scores)[::-1]

    # Exclude the given movie itself
    similar_movie_indices = similar_movie_indices[similar_movie_indices != movie_id - 1]

    # Get top n similar movie indices
    top_similar_indices = similar_movie_indices[:n]

    return top_similar_indices

def recommend_movies(user_id, n=5):
    user_ratings = user_movie_ratings[user_id - 1]

    # Initialize a dictionary to store aggregated ratings for each movie
    aggregated_ratings = {}

    # Loop through each movie the user hasn't rated
    for movie_id, rating in enumerate(user_ratings):
        if rating == 0:
            # Get top similar movies for the unrated movie
            top_similar_indices = get_top_similar_items(movie_id + 1)

            # Aggregate ratings for each similar movie
            for similar_movie_id in top_similar_indices:
                if similar_movie_id not in aggregated_ratings:
                    aggregated_ratings[similar_movie_id] = 0
                # Add the similarity-weighted rating to the aggregated rating
                aggregated_ratings[similar_movie_id] += user_movie_ratings[user_id - 1][similar_movie_id] * \
                                                        item_similarity_matrix[movie_id][similar_movie_id]

    # Sort aggregated ratings in descending order
    sorted_aggregated_ratings = sorted(aggregated_ratings.items(), key=lambda x: x[1], reverse=True)

    # Get top n recommended movies
    top_recommendations = [movie_id + 1 for movie_id, _ in sorted_aggregated_ratings[:n]]

    return top_recommendations

# Example usage
user_id = 2  # Assuming user ID 1
recommended_movies = recommend_movies(user_id)
print("Recommended movies for user", user_id)
for movie_id in recommended_movies:
    print("Movie ID:", movie_id)
