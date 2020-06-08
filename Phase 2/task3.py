from os import listdir
import sys
import numpy as np
import math
import operator
import re
import scipy
from scipy import linalg
import operator
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import xml.etree.ElementTree as et

def SVD(object_feature_matrix):
    
    U, S, VT = scipy.linalg.svd(object_feature_matrix.astype(np.float64), full_matrices = False)
    object_topic_matrix = U
    topic_feature_matrix = VT

    return object_topic_matrix, topic_feature_matrix

def PCA(object_feature_matrix, k):

    pca                           = sklearn.decomposition.PCA(copy = True, iterated_power = 'auto', n_components = k, random_state = None,  svd_solver = 'auto')
    reduced_object_feature_matrix = pca.fit_transform(object_feature_matrix.astype(np.float64))
    topic_feature_matrix          = pca.components_

    return reduced_object_feature_matrix, topic_feature_matrix, pca.singular_values_

def LDA(object_feature_matrix, k, model):

    if model == "CM" or model == "CM3x3":
        scaled_data_mat = minmax_scale(object_feature_matrix, feature_range = (0,5), axis = 1)
    else:
        scaled_data_mat = object_feature_matrix

    lda_model          = LatentDirichletAllocation(k, learning_method = 'online', learning_offset = 100., random_state = 0)
    lda_model.fit(scaled_data_mat)

    reduced_object_feature_matrix      = lda_model.transform(scaled_data_mat)
    topic_feature_matrix  = lda_model.components_

    return reduced_object_feature_matrix, topic_feature_matrix


def get_locationid(location_name):

    tree = et.parse('./devset_topics.xml')
    root = tree.getroot()

    for child in root:
        if (child[1].text) == location_name:
            location_id = int(child[0].text)
            break

    return location_id


def get_locationid_list():
    tree = et.parse('./devset_topics.xml')
    root = tree.getroot()

    location_ids = []
    for child in root:
        location_ids.append(int(child[0].text))

    return location_ids


def create_object_feature_matrix(model):

    no_photoids           = 0
    global_image_ids      = list()
    color_feature_list = []
    location_imageid_tuple = []
    for file in listdir("./descvis/img/"):
        match = re.match("(.*)\s(.*).csv", file)
        if match:
            if match.group(2) == model:
                file_name = "./descvis/img/" + file
                input_file = open(file_name)

                location_name = match.group(1)
                location_id   = get_locationid(location_name)
                
                for line in input_file:
                    no_photoids   = no_photoids + 1
                    image_object  = line.split(",")
                    global_image_ids.append(int(image_object[0]))

                    location_imageid_tuple.append((location_id, int(image_object[0])))

                    color_feature = image_object[1:]
                    
                    dimensions    = len(color_feature)

                    color_feature_list.append(color_feature)                     

    
    object_feature_matrix = np.asarray(color_feature_list).reshape((no_photoids, dimensions)).astype(np.float64)
    
    return object_feature_matrix, no_photoids, global_image_ids, location_imageid_tuple                

def getimagedistances(object_feature_matrix, query_imageid, global_image_ids, no_photoids):

    query_index = global_image_ids.index(query_imageid)
    query = object_feature_matrix[query_index,:]
    imageid_distance_tuples = []

    i = 0
    while i < no_photoids:
        if i != query_index:
            image = object_feature_matrix[i,:]
            dist = np.linalg.norm(np.subtract(query, image)) #L2 norm
            imageid_distance_tuples.append((global_image_ids[i], dist))
        i += 1

    imageid_distance_tuples = sorted(imageid_distance_tuples, key = operator.itemgetter(1))

    return imageid_distance_tuples

def getlocationscores(location_imageid_tuple, imageid_distance_tuples, query_location_id):

    locationid_score_tuples   = []
    location_ids = get_locationid_list()

    for loc_id in location_ids:
        total_score = 0
        if loc_id != query_location_id:
            #Get all images ids for a location id
            img_ids = [item[1] for item in location_imageid_tuple if item[0] == loc_id]
            no_img_ids = len(img_ids)
            for i in img_ids:
                x = [item[1] for item in imageid_distance_tuples if item[0] == i][0]
                matching_score = 1 / (1 + x)
                total_score += matching_score

            locationid_score_tuples.append((loc_id, total_score / no_img_ids))

    locationid_score_tuples = sorted(locationid_score_tuples, key = operator.itemgetter(1), reverse = True)

    return locationid_score_tuples

def print_latent_semantics(projected_feature_topic_matrix, k):

    print("Printing Latent Semantics")
    for i in range(0, k):
        print("Latent Semantic: {}".format(i + 1))
        print(projected_feature_topic_matrix[:,i])


np.set_printoptions(threshold = np.nan)
k                     = int(sys.argv[1])
model                 = sys.argv[2]
feature_decomposition = sys.argv[3]
query_imageid         = int(sys.argv[4])

object_feature_matrix, no_photoids, global_image_ids, location_imageid_tuple = create_object_feature_matrix(model)

if feature_decomposition == "LDA":

    reduced_object_feature_matrix, topic_feature_matrix = LDA(object_feature_matrix, k, model)

    print_latent_semantics(topic_feature_matrix.transpose(), k)

elif feature_decomposition == "SVD":


    object_topic_matrix, topic_feature_matrix = SVD(object_feature_matrix)
    
    T = topic_feature_matrix.transpose()
    feature_Ktopic_matrix = T[:,:k] 
    reduced_object_feature_matrix = np.dot(object_feature_matrix.astype(np.float64), feature_Ktopic_matrix.astype(np.float64))

    object_Ktopic_matrix = object_topic_matrix[:,:k]

    T = object_feature_matrix.transpose()
    projected_feature_topic_matrix = np.dot(T.astype(np.float64), object_Ktopic_matrix)
    
    print_latent_semantics(projected_feature_topic_matrix, k)
    
 
elif feature_decomposition == "PCA":
    reduced_object_feature_matrix, topic_feature_matrix, singular_values = PCA(object_feature_matrix, k)

    matrix = np.dot(np.diag(singular_values), topic_feature_matrix)
    projected_feature_topic_matrix = matrix.transpose()
    print_latent_semantics(projected_feature_topic_matrix, k)

imageid_distance_tuples = getimagedistances(reduced_object_feature_matrix, query_imageid, global_image_ids, no_photoids)

print("Query Image ID:{}".format(query_imageid))
print("Top 5 related images:")
i = 0
for (imageid, dist) in imageid_distance_tuples:
    print("Image ID:{} Matching Score:{}".format(imageid, 1/(dist + 1)))
    i += 1
    if i == 5:
        break
query_location_id, = [item[0] for item in location_imageid_tuple if item[1] == query_imageid]
print("Query location ID:{}".format(query_location_id))
print ("Top 5 related locations:")

locationid_score_tuples = getlocationscores(location_imageid_tuple, imageid_distance_tuples, query_location_id)

i = 0
for (loc_id, score) in locationid_score_tuples:
    print("Location ID: {} Matching Score:{}".format(loc_id, score))
    i += 1
    if i == 5:
        break

                

