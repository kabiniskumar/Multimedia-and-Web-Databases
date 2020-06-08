import numpy as np
import math
from os import listdir
import re
import xml.etree.ElementTree as et
import pprint
import pickle
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
import sys
from pymongo import MongoClient


def object_feature_matrix_model(loc, model, image_store):

    image_set = list(db.locations_images.find({"number": loc, "model": model}))
    m = len(image_set)
    detail = image_set[0]
    n = len(detail["details"]["values"])
    A_model = np.zeros((m, n))
    for each in image_set:
        details = each["details"]
        index = image_store.index(details["image_id"])
        A_model[index,:] = details["values"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(A_model)
########################################################################################################
# Creates an object-feature matrix for the all images and all descriptor models for a given location id
########################################################################################################
def object_feature_matrix(loc):

    m = db.locations_images.count_documents({"number": loc, "model": "CM3x3"})
    n = model_index[len(model_index) - 1]
    np.set_printoptions(suppress=True)
    A = np.zeros((m, n))
    image_set = db.locations_images.find({"number": loc, "model": "CM3x3"})
    image_store = []
    i = 0
    for each in image_set:
        details = each["details"]

        if details["image_id"] not in image_store:
            image_store.append(details["image_id"])
            A[i][model_index[0]:model_index[1]] = details["values"]
            i += 1
        else:
            index = image_store.index(details["image_id"])
            A[index][model_index[0]:model_index[1]] = details["values"]

    for i in range(1, len(models)):
        model_sclaed = object_feature_matrix_model(loc, models[i], image_store)
        ind = models.index(models[i])
        for j in range(len(model_sclaed)):
            A[j][model_index[ind]:model_index[ind+1]] = model_sclaed[j]

    return A, image_store

#####################################################
# Returns object feature matrix for all images
#######################################################
def compute_image_feature_all():

    m = db.locations_images.count_documents({"model": "CM"})
    n = model_index[len(model_index) - 1]

    # Combines the object-feature matrix computed for each location into a single matrix

    all_image_store = []
    matrix = np.zeros((m, n))

    i = start = 0
    for each in mapping_loc:
        temp_mat, img_list = object_feature_matrix(each)
        all_image_store.extend(img_list)
        end_ind = len(temp_mat)

        for j in range(len(temp_mat)):
            matrix[i] = temp_mat[j]
            i += 1

        start = start+end_ind
    return matrix, all_image_store

########################################################
# Tries to load img_pickle or creates object feature
# matrix if failure occurs
#######################################################
def create_or_load_obj_feature_mat(img_pickle):

    try:
        img_dict = pickle.load(open(img_pickle, "rb"))
    except (OSError, IOError) as e:
        obj_feature_mat, img_list = compute_image_feature_all()

        img_dict = {}
        img_dict["obj_feature_mat"] = obj_feature_mat
        img_dict["imglist"] = img_list

        outfile = open(img_pickle, 'wb')
        pickle.dump(img_dict, outfile)
        outfile.close()
    else:
        img_list = img_dict["imglist"]
        obj_feature_mat = img_dict["obj_feature_mat"]

    return obj_feature_mat, img_list

##########################################
# Pickles LSH index structure
############################################
def pickle_LSH(LSH_hash_table, LSH_bucket):

    filename = "LSH_index"
    out_file = open(filename,'wb')
    pickle.dump(LSH_hash_table, out_file)
    out_file.close()

    filename = "LSH_bucket"
    out_file = open(filename,'wb')
    pickle.dump(LSH_bucket, out_file)
    out_file.close()

#####################################################
# Lower level Hash function
# Gives bucket ID for a projection, b and w
####################################################
def hash_func(projection, b , w):

    bucketID = math.floor((projection + b) / w)

    return bucketID

#################################################################
# Main LSH function, takes set of vectors and creates
# index structure for the input vectors and a given L and K.
##################################################################
def LSH(input_vec, global_image_ids, L, K):

    # Dictionary of hashes, each key corresponds to a layer of hash functions
    L_hashes = {}
    # Dictionary of buckets, each key corresponds to a layer of hash functions
    L_buckets = {}
    # Dictionary of hash functions/dictionaries in a layer
    K_hashes = {}
    # Dictionary of bucket dictionaries in a layer 
    K_buckets = {}
    # Dictionary storing bucket ID of each imageID
    image_hash  = {}
    # Dictionary storing image IDs preset in a bucketID
    bucket = {}

    n_images, n_dimensions = input_vec.shape
    X_unit_vec = np.zeros((K,n_dimensions))
    # List of projections in a layer
    projections = np.zeros((K,n_images))

    for z in range(L):

        K_hashes.clear()
        K_buckets.clear()

        b = np.random.uniform(0, w, (K,1))

        for j in range(K):

            X_unit_vec = np.zeros((1,n_dimensions))
            X = np.random.normal(0, 1, (1,n_dimensions))
            norm = np.linalg.norm(X)
            X_unit_vec = X/norm

            # Clear image_hash and bucket dicts for new hash function
            image_hash.clear()
            bucket.clear()

            for i in range(n_images):
                X_unit_vec = X/norm
                projections[j][i] = np.dot(input_vec[i], X_unit_vec.T)
                bucketID = hash_func(projections[j][i], b[j][0], w)
                #print("imgID:{} bucketID:{} projection:{}".format(global_image_ids[i],bucketID,projections[j][i]))
                image_hash[global_image_ids[i]] = bucketID
                if bucketID in bucket:
                    bucket[bucketID].append(global_image_ids[i])
                else:
                    bucket[bucketID] = [global_image_ids[i]]

                #print("Layer:{},K:{},bucket_dict:{}".format(z,j,bucket))

            K_hashes["K" + str(j)] = image_hash.copy()
            K_buckets["K" + str(j)] = bucket.copy()

        L_hashes["L" + str(z)] = K_hashes.copy()
        L_buckets["L" + str(z)] = K_buckets.copy()

    return L_hashes, L_buckets, projections

######################################################
# Execution Starts Here
######################################################
L = int(sys.argv[1])
K = int(sys.argv[2])

# Width of bucket
w = 1.5

client = MongoClient()
db = client.MWDB_devset

models = ["CM3x3", "CN3x3", "CSD", "GLRLM3x3", "HOG", "LBP3x3"]
model_index = [0, 81, 180, 244, 640, 721, 865]
mapping_loc = db.locations_images.distinct('number')

img_pickle = "img_dict"

#Get object feature matrix of images
obj_feature_mat, global_image_ids = create_or_load_obj_feature_mat(img_pickle)

# Create LSH
L_hashes, L_buckets, projections = LSH(obj_feature_mat, global_image_ids, L, K)

# Pickle LSH index structure for future use
pickle_LSH(L_hashes, L_buckets)