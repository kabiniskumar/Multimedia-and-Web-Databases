import numpy
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms, squared_norm
import pickle
from numpy import linalg as LA
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from collections import defaultdict
from flask import Flask, render_template

app = Flask(__name__)

random_state = check_random_state(None)
clusters = defaultdict(list)
groups = defaultdict(list)


@app.route("/")
def visualise_top_images():
    return render_template('kmeans_clusters.html', clusters=clusters)


def spectral(matrix, k):
    matrix = matrix.todense()
    matrix[matrix == 1] = -1
    sum_row = abs(matrix[0].sum())
    for i in range(len(matrix)):
        matrix[i, i] = sum_row
    matrix = sparse.csr_matrix(matrix, dtype=float)
    C, D = sparse.linalg.eigsh(matrix, k, which='SM')
    D = sparse.csr_matrix(D, dtype=float)
    labels_spectral = kmeans(D, k)
    return labels_spectral


def inner_k_means(X, centers, sqared_norms, samples):
    X_indptr = X.indptr
    X_data = X.data
    X_indices = X.indices
    labels = []
    for i in range(samples):
        labels.append(0)
    inertia = 0.0
    center_squared_norms = numpy.zeros(k)
    for c_index in range(k):
        center_squared_norms[c_index] = LA.norm(centers[c_index]) ** 2
    for sample_index in range(samples):
        min_dist = -1
        for c_index in range(k):
            dist = 0.0
            for j in range(X_indptr[sample_index], X_indptr[sample_index + 1]):
                dist += centers[c_index, X_indices[j]] * X_data[j]
            dist *= -2
            dist += center_squared_norms[c_index]
            dist += sqared_norms[sample_index]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[sample_index] = c_index
        inertia += min_dist
    return inertia, labels


def assign_labels(X, squared_norms, centers, distances, samples):
    labels = numpy.full(samples, -1, numpy.int32)
    if distances is None:
        distances = numpy.zeros(shape=(X.shape[0],), dtype=X.dtype)
    inertia, labels = inner_k_means(X, centers, squared_norms, samples)
    return labels, inertia


def maximization(X, labels, distances, centers, samples):
    X_indptr = X.indptr
    X_data = X.data
    X_indices = X.indices
    sameple_weight = numpy.ones(samples, dtype=X.dtype)
    weight_cluster = numpy.zeros(k, dtype=float)
    for i in range(samples):
        c = labels[i]
        weight_cluster[c] += sameple_weight[i]
        empty_clusters = numpy.where(weight_cluster == 0)[0]
    n_empty_clusters = empty_clusters.shape[0]
    if n_empty_clusters > 0:
        far_points = distances.argsort()[::-1][:n_empty_clusters]

        assign_rows_csr(X, far_points.astype(numpy.intp), empty_clusters.astype(numpy.intp), centers)
        for i in range(n_empty_clusters):
            weight_cluster[empty_clusters[i]] = 1

    for i in range(len(labels)):
        curr_label = labels[i]
        for index in range(X_indptr[i], X_indptr[i + 1]):
            j = X_indices[index]
            centers[curr_label, j] += X_data[index] * sameple_weight[i]
    numpy.true_divide(centers, weight_cluster[:, numpy.newaxis], out=centers, casting="unsafe")
    return centers


def kmeans(X, k):
    samples = X.shape[0]
    best_inertia = None
    best_labels = None
    x_squared_norms = row_norms(X, squared=True)
    seeds = random_state.permutation(samples)[:k]
    centers = X[seeds]
    centers = centers.toarray()
    distances = numpy.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # Iterations
    for i in range(100):
        centers_old = centers.copy()
        labels, inertia = assign_labels(X, x_squared_norms, centers, distances, samples)
        centers = maximization(X, labels, distances, centers, samples)
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_inertia = inertia

        center_shift = squared_norm(centers_old - centers)

    if center_shift > 0:
        best_labels, best_inertia = assign_labels(X, x_squared_norms, centers, distances, samples)
    return best_labels


infile = open('Input_graph_final', 'rb')
img_graph = pickle.load(infile)

# Retrieving image store
img_store_file = open('image_store100', 'rb')
img_store = pickle.load(img_store_file)

X = img_graph

input_algorithm = input("Please enter 1 for K Means and 2 for Spectral Clustering ")

k = int(input("Enter the value of k: "))

if int(input_algorithm) == 1:
    labels = kmeans(X, k)
else:
    labels = spectral(X, k)

dictionary = dict(zip(img_store, labels))

for key, value in dictionary.items():
    if key not in clusters:
        clusters[value].append(key)

for key, value in clusters.items():
    value = [s + '.jpg' for s in value]
    clusters[key] = value
app.run()
