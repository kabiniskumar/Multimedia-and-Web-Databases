from pymongo import MongoClient
import time
import numpy as np
from scipy.linalg import svd
from scipy import dot


# Mongo db connection
connection = MongoClient('localhost', 27017)
db = connection.MWDB_devset

# Input k from the user
k = int(input("Enter the value of k: "))

# get all the location details from the database
all_locations_details = list(db.locations.find())

all_terms = []
all_locations = []

# create array of 'all_locations' and 'all_terms'
for item in all_locations_details:
    if item['locationQuery'] is not all_locations:
        all_locations.append(item['locationQuery'])
    for detail in item['details']:
        if detail['term'] is not all_terms:
            all_terms.append(detail['term'])

# create zeros matrix of length of all locations and all terms
location_data_matrix = np.zeros(shape=(len(all_locations_details), len(all_terms)))

# create locations - terms matrix
for location in all_locations_details:
    for detail in location['details']:
        location_data_matrix[all_locations.index(location['locationQuery'])][all_terms.index(detail['term'])] = detail['TF-IDF']

# transpose of location - term matrix
location_data_matrix_transpose = location_data_matrix.transpose()

location_location_matrix = dot(location_data_matrix, location_data_matrix_transpose)

# SVD
U, s, VT = svd(location_location_matrix)

# project original matrix to the new location-semantics matrix
project_matrix = dot(location_location_matrix, U)

features_locations_matrix = project_matrix.transpose()

# Get top k- latent semantics
top_k_latent_semantics = features_locations_matrix[:k, :]
# d = {}
count = 1
for item in top_k_latent_semantics:
    d = {}
    print('\n')
    print("Latent Semantic", count)
    print()
    count += 1
    i = 0
    while i < len(all_locations):
        d[all_locations[i]] = item[i]
        i += 1
    for w in sorted(d, key=d.get, reverse=True):
        print(w, ':', d[w])
