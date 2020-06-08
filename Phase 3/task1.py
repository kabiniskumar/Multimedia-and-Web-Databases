from pymongo import MongoClient
import numpy
from scipy.spatial import distance
import sklearn.preprocessing as preprocessing
import gc
from scipy import sparse
import pickle
from collections import defaultdict


# Connecting to Mongo DB
client = MongoClient()

# Get the database
db = client.DEVSET


# Creates an object-feature matrix for the all images and all descriptor models for a given location id
def object_feature_matrix(loc):

    m = db.locations_images.count_documents({"number": loc, "model": "CM3x3"})
    n = model_index[len(model_index) - 1]
    numpy.set_printoptions(suppress=True)
    A = numpy.zeros((m, n))
    image_set = list(db.locations_images.find({"number": loc, "model": "CM3x3"}))
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

    # A[: ,model_index[0]:model_index[1]] = scaler.fit_transform(A[: ,model_index[0]:model_index[1]])
    X = A[:, model_index[0]:model_index[1]]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    A[:, model_index[0]:model_index[1]] = scaler.fit_transform(X.reshape(X.shape[1],X.shape[0])).reshape(X.shape[0],X.shape[1])

    for i in range(1, len(models)):
        model_sclaed = object_feature_matrix_model(loc, models[i], image_store)
        ind = models.index(models[i])
        for j in range(len(model_sclaed)):
            A[j][model_index[ind]:model_index[ind+1]] = model_sclaed[j]

    return A, image_store


def object_feature_matrix_model(loc, model, image_store):

    image_set = list(db.locations_images.find({"number": loc, "model": model}))
    m = len(image_set)
    detail = image_set[0]
    n = len(detail["details"]["values"])
    A_model = numpy.zeros((m, n))
    for each in image_set:
        details = each["details"]
        index = image_store.index(details["image_id"])
        A_model[index,:] = details["values"]

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    return scaler.fit_transform(A_model.reshape(A_model.shape[1],A_model.shape[0])).reshape(A_model.shape[0],A_model.shape[1])


def compute_image_feature_all():

    m = db.locations_images.count_documents({"model": "CM3x3"})
    n = model_index[len(model_index) - 1]

    # Combines the object-feature matrix computed for each location into a single matrix

    all_image_store = []
    matrix = numpy.zeros((m, n))

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


def compute_similarity(P):

    gc.collect()
    # Uses Euclidean distance to compute similarity
    dist_matrix = distance.cdist(P, P, 'euclidean')

    return dist_matrix


def create_graph(img_sim, k):

    graph_out = defaultdict(list)
    for i in range(len(img_sim)):
        top_k = numpy.argsort(img_sim[i])[:k+1].tolist()

        sim_images_list = []
        for each in top_k:
            if each != i:
                sim_images_list.append(img_store[each])

        graph_out[img_store[i]] = sim_images_list

        for j in range(len(img_sim)):
            if j in top_k and j != i:
                img_sim[i, j] = 1
            else:
                img_sim[i, j] = 0

    img_graph = sparse.csr_matrix(img_sim, dtype=numpy.float)
    outfile = open('Input_graph_final', 'ab')
    pickle.dump(img_graph, outfile)
    outfile.close()
    print(graph_out)


models = ["CM3x3", "CN3x3", "CSD", "GLRLM3x3", "HOG", "LBP3x3"]
model_index = [0, 81, 180, 244, 640, 721, 865]
k = int(input("Enter the value of k: "))

scaler = preprocessing.StandardScaler()


mapping_loc = db.locations_images.distinct('number')
obj_feature, img_store = compute_image_feature_all()
img_img = compute_similarity(obj_feature)

create_graph(img_img, k)
img_outfile = open('image_store100', 'ab')
pickle.dump(img_store, img_outfile)
img_outfile.close()
