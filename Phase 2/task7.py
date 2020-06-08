def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from pymongo import MongoClient
import numpy as np
from tensorly.decomposition import parafac
from collections import defaultdict
from sklearn.cluster import KMeans
import gc

connection = MongoClient('localhost', 27017)
db = connection.MWDB_devset

k = int(input("Enter the value of k "))

all_location_details = list(db.locations.find())
all_user_details = list(db.users.find())
all_image_details = list(db.images.find())

all_locations = []
all_images = []
all_users = []
all_locations_terms = {}
all_images_terms = {}
all_users_terms = {}

for item in all_location_details:
    terms = []
    if item['locationQuery'] is not all_locations:
        all_locations.append(item['locationQuery'])
    for data in item['details']:
        terms.append(data['term'])
        all_locations_terms[item['locationQuery']] = terms


for item in all_user_details:
    terms = []
    if item['userID'] is not all_users:
        all_users.append(item['userID'])
    for data in item['details']:
        terms.append(data['term'])
    all_users_terms[item['userID']] = terms

for item in all_image_details:
    terms = []
    # all_images_terms['imageid'] = item['imageID']
    if item['imageID'] is not all_images:
        all_images.append(item['imageID'])
    for data in item['details']:
        terms.append(data['term'])
    all_images_terms[item['imageID']] = terms


user_groups = defaultdict(list)
image_groups = defaultdict(list)
loc_groups = defaultdict(list)

def get_non_overlapping_groups1(factor_matrix, k, group, all_items):
    global key, value
    # Number of clusters
    kmeans = KMeans(n_clusters=int(k))
    # Fitting the input data
    kmeans = kmeans.fit(factor_matrix)
    # Getting the cluster labels
    labels = kmeans.predict(factor_matrix)
    dictionary = dict(zip(all_items, labels))
    for key, value in dictionary.items():
        if key not in user_groups:
            user_groups[value].append(key)


def get_non_overlapping_groups2(factor_matrix, k, group, all_items):
    global key, value
    # Number of clusters
    kmeans = KMeans(n_clusters=int(k))
    # Fitting the input data
    kmeans = kmeans.fit(factor_matrix)
    # Getting the cluster labels
    labels = kmeans.predict(factor_matrix)
    dictionary = dict(zip(all_items, labels))
    for key, value in dictionary.items():
        flag = 0
        for i in range(1, k+1):
            li = image_groups[i]
            if key in li:
                flag = 1
                break

        if flag == 0 and key not in image_groups and len(image_groups[value]) < div_img:
            # loc_groups[value].append(key)
            image_groups[value].append(key)

div_loc = int(len(all_locations)/k)
div_img = int(len(all_images)/k)

div_usr = int(len(all_users)/k)

def get_non_overlapping_groups3(factor_matrix, k, group, all_items):
    global key, value
    # Number of clusters
    kmeans = KMeans(n_clusters=int(k))
    # Fitting the input data
    kmeans = kmeans.fit(factor_matrix)
    # Getting the cluster labels
    labels = kmeans.predict(factor_matrix)
    dictionary = dict(zip(all_items, labels))
    for key, value in dictionary.items():
        flag = 0
        for i in range(1, k + 1):
            li = loc_groups[i]
            if key in li:
                flag = 1
                break

        if flag == 0 and key not in loc_groups and len(loc_groups[value]) < div_loc:
            # loc_groups[value].append(key)
            loc_groups[value].append(key)

def paradecomp(i):

    gc.collect()
    path = "test"+str(i)+".npy"
    tensor1 = np.load(path)
    factors = parafac(tensor1, rank=int(k), init='random')
    return factors


for i in range(1,9):
    factors = paradecomp(i)
    users_factor_matrix = factors[0]
    images_factor_matrix = factors[1]
    locations_factor_matrix = factors[2]

    get_non_overlapping_groups2(images_factor_matrix, k, "Images", all_images)


factors = paradecomp(1)
users_factor_matrix = factors[0]
images_factor_matrix = factors[1]
locations_factor_matrix = factors[2]
get_non_overlapping_groups1(users_factor_matrix, k, "Users", all_users)
get_non_overlapping_groups3(locations_factor_matrix, k, "Locations", all_locations)
for key, value in user_groups.items():
    print("Group Users ", key + 1, value)
    print()
print(len(user_groups))
for key, value in image_groups.items():
    if key < k:
        print("Group Images ", key + 1, value)
        print()
print(len(image_groups))
for key, value in loc_groups.items():
    if key < k:
        print("Group Locations ", key + 1, value)
        print()
print(len(loc_groups))


