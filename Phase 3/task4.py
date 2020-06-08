from pymongo import MongoClient
import numpy
import time
import pickle
from scipy import sparse
from flask import Flask, render_template

app = Flask(__name__)

# Connecting to Mongo DB
client = MongoClient()

# Get the database
db = client.MWDB_devset
numpy.set_printoptions(suppress=True, linewidth=100)


def pagerank(graph, seed, alpha=0.85, maxerr=0.001):
    graphsum = graph.sum(axis=0)
    graphsum[graphsum == 0] = 1

    graph /= graphsum
    data = sparse.csr_matrix(graph)

    n = len(img_store)
    ro, r = numpy.zeros(n), numpy.ones(n)
    r = r/n

    # Setting teleport vector values
    teleport = numpy.zeros(n)
    for each in seed:
        teleport[each] = 1
    teleport = teleport / sum(teleport)


    while numpy.sum(numpy.abs(r - ro)) > maxerr :
        ro = r.copy()
        for i in range(n):
            links = numpy.array(data[:, i].todense())[:, 0]
            r[i] = ro.dot(links * alpha) + teleport[i] * (1 - alpha)

    return r / sum(r)


def compute_seed(img_list):
    seed_vector = []

    for each in img_list:
        index = img_store.index(each)
        seed_vector.append(index)

    return seed_vector


top_images = []


@app.route("/")
def visualize_images():
    return render_template('ppr.html', top_images=top_images)


def get_top_images(scores, n):
    indices = numpy.argsort(scores)[::-1][:n].tolist()
    images = pickle.load(open("image_store100", "rb"))

    for index in indices:
        top_images.append(images[index] + '.jpg')
    print(top_images)
    app.run()


start = time.time()


# Retrieving graph
infile = open("Input_graph_final", 'rb')
img_graph = pickle.load(infile)

# Retrieving image store
img_store_file = open("image_store100", 'rb')
img_store = pickle.load(img_store_file)

# Compute seed vector
images_input = input("Enter the image IDs: ")
images_li = images_input.split(" ")
K = int(input("Enter the value of k: "))
seed = compute_seed(images_li)

# Compute personalized pagerank
pr = pagerank(img_graph, seed)

# visualize_images(pr, K)
get_top_images(pr, K)

# 4720907573 10659727256 7754466396 7441611662