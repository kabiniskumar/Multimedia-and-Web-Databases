from pymongo import MongoClient
import numpy
from collections import OrderedDict
import itertools
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import LatentDirichletAllocation

# Compute top k latent semantics corresponding to the given location
# Compute the top 5 related location to the given location based on the latent semantics

# Connecting to Mongo DB
client = MongoClient()

# Get the database
db = client.MWDB_devset


# Creates an object-feature matrix for the all images and all descriptor models for a given location id
def create_obj_feature_matrix(loc):

    m = db.locations_images.count_documents({"number": loc, "model": "CM"})
    n = model_index[len(model_index)-1]
    numpy.set_printoptions(suppress=True)
    A = numpy.zeros((m, n))
    image_store = []

    image_set = db.locations_images.find({"number": loc})
    i = 0
    for each in image_set:
        details = each["details"]
        ind = models.index(each["model"])

        if details["image_id"] not in image_store:
            image_store.append(details["image_id"])
            A[i][model_index[ind]:model_index[ind+1]] = details["values"]
            i += 1
        else:
            index = image_store.index(details["image_id"])
            A[index][model_index[ind]:model_index[ind + 1]] = details["values"]

    return A, image_store


def find_latent_semantics_svd(A, k):

    # Computes svd on matrix A with svd_comp representing the principal axes and latent_comp gives latent semantics
    svd.fit(A)
    new_A = svd.transform(A)
    svd_comp = svd.components_
    latent_comp = numpy.dot(numpy.diag(svd.singular_values_), svd_comp)
    print("The top ", k, " Latent Semantics are: ")
    for i in range(len(latent_comp)):
        print("Latent semantic ", i+1)
        print(latent_comp[i])


def find_latent_semantics_pca(A, k):

    # StandardScaler() normalizes the mean and mean centers the data
    # Computes pca on matrix A with p_comp representing the principal axes and latent_comp gives latent semantics
    A_temp = A
    A_std = StandardScaler().fit_transform(A)
    pca.fit(A_std)
    p_comp = pca.components_
    latent_comp = numpy.dot(numpy.diag(pca.singular_values_), p_comp)
    scale = StandardScaler().fit(A_temp)
    mean_reversed = scale.inverse_transform(latent_comp)
    print("The top ", k, " Latent Semantics are: ")
    for i in range(len(mean_reversed)):
        print("Latent semantic ", i + 1)
        print(mean_reversed[i])


def find_latent_semantics_lda(A, k):

    # Computes lda on matrix A with lda_comp representing the latent semantics
    A_scaled = minmax_scale(A)
    lda.fit(A_scaled)
    X = lda.transform(A_scaled)
    lda_comp = lda.components_

    print("The top ", k, "Latent Semantics are: ")
    for i in range(len(lda_comp)):
        print("Latent semantic ", i + 1)
        print(lda_comp[i])


def compute_image_feature_all(algo):

    img_loc_count = db.locations_images.count_documents({"number": location_id, "model": "CM"})

    m = db.locations_images.count_documents({"model": "CM"}) - img_loc_count
    n = model_index[len(model_index) - 1]

    # Combines the object-feature matrix computed for each location into a single matrix

    all_image_store = []
    matrix = numpy.zeros((m, n))

    i = start = 0
    for each in mapping_loc:

        if each != location_id:
            temp_mat, img_list = create_obj_feature_matrix(each)
            all_image_store.extend(img_list)
            end_ind = len(temp_mat)

            for j in range(len(temp_mat)):
                matrix[i] = temp_mat[j]
                i += 1

            start = start+end_ind

    # Transforms the query matrix and the data matrix to the new dimensions of the subspace
    # Computes similarity of the points in the new subspace
    if algo == 'SVD':
        query = svd.transform(A)
        all_objects = svd.transform(matrix)
        img_img = compute_similarity(query, all_objects)
        compute_rank(img_img, all_image_store)
    elif algo == 'PCA':
        query_std = StandardScaler().fit_transform(A)
        matrix_std = StandardScaler().fit_transform(matrix)
        query = pca.transform(query_std)
        all_objects = pca.transform(matrix_std)
        img_img = compute_similarity(query, all_objects)
        compute_rank(img_img, all_image_store)
    elif algo == 'LDA':
        query_scaled = minmax_scale(A)
        matrix_scaled = minmax_scale(matrix)
        query = lda.transform(query_scaled)
        all_objects = lda.transform(matrix_scaled)
        img_img = compute_similarity(query, all_objects)
        compute_rank(img_img, all_image_store)


def compute_similarity(P, Q):

    # Uses Euclidean distance to compute similarity
    dist_matrix = numpy.ones((len(P), len(Q)))

    for i in range(len(P)):
        for j in range(len(Q)):
            dist = distance.euclidean(P[i], Q[j])
            dist_matrix[i, j] = dist

    return dist_matrix


def compute_rank(matrix, all_image_store):

    img_list = []
    loc_list = []
    img_loc = db.locations_images.find({"model": "CM"}, {"number": 1, "details.image_id": 2, "_id": 0})
    for each in img_loc:
        details = each["details"]
        img_list.append(details["image_id"])
        loc_list.append(each["number"])

    top_similar_img = []

    # Finds the top 5 similar images for all the images in the given location
    for i in range(len(matrix)):
        max_5 = numpy.argsort(matrix[i])[:5].tolist()
        for val in max_5:
            top_similar_img.append(all_image_store[val])

    # Aggregates the similar images based on their respective locations
    top_location_count = {}
    for each in top_similar_img:
        loc = loc_list[img_list.index(each)]
        if loc not in top_location_count:
            top_location_count[loc] = 1
        else:
            top_location_count[loc] += 1

    print_top5_loc(top_location_count)


def print_top5_loc(loc_dict):

    # Displays the top 5 similar locations
    od = OrderedDict(sorted(loc_dict.items(), key=lambda x: x[1], reverse=True))
    top_n = itertools.islice(od.items(), 0, 5)
    print("Top 5 locations are ")
    for key, value in top_n:
        print("Location ID: ", key, "Similarity Score: ", value)


args = input("Enter LocationID, k and Dimensionality Reduction Algorithm : ")
location_id, top, alg = args.split(" ")
location_id = int(location_id)
k = int(top)


models = ["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]
model_index = [0, 9, 90, 101, 200, 264, 308, 704, 785, 801, 945]
mapping_loc = db.locations_images.distinct('number')

# Creating an object-feature matrix for the images from the given location
A, store = create_obj_feature_matrix(location_id)


if alg == 'SVD':
    svd = TruncatedSVD(n_components=k)
    find_latent_semantics_svd(A, k)
    compute_image_feature_all(alg)
elif alg == 'PCA':
    pca = PCA(n_components=k)
    find_latent_semantics_pca(A, k)
    compute_image_feature_all(alg)
elif alg == 'LDA':
    lda = LatentDirichletAllocation(n_components=k, max_iter=10, learning_method='online', random_state=0)
    find_latent_semantics_lda(A, k)
    compute_image_feature_all(alg)


