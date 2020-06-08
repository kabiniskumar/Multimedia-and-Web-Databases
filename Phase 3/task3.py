import pickle
import numpy as np
from scipy.sparse import csr_matrix
from flask import Flask, render_template

app = Flask(__name__)
top_images = []


# Compute page rank
def page_rank(graph, d=.85, maxerr=.001):
    n = graph.shape[0]
    count = 0

    # transform G into stochastic or markov matrix graph
    col_sum = graph.sum(axis=0)
    col_sum[col_sum == 0] = 1
    graph /= col_sum
    a = csr_matrix(graph)

    ro, r = np.zeros(n), np.ones(n)
    r = r / n
    teleportation = np.ones(n) / float(n)
    while np.sum(np.abs(r - ro)) > maxerr:
        ro = r.copy()
        # iterative method
        for i in range(0, n):
            incoming_links = np.array(a[:, i].todense())[:, 0]
            r[i] = ro.dot(incoming_links * d) + teleportation[i] * (1 - d)
        count += 1
    print("Iterations", count)
    return r / sum(r)


@app.route("/")
def visualise_top_images():
    return render_template('pr.html', top_images=top_images)


K = int(input("Enter the value of k: "))


# Get top K dominant images using Page rank
def get_top_images(pr, K):
    # indices = pr.argsort()[-K:][::-1]
    indices = np.argsort(pr)[::-1][:K].tolist()
    images = pickle.load(open("image_store100", "rb"))

    for index in indices:
        top_images.append(images[index] + '.jpg')
    print(top_images)
    app.run()


G = pickle.load(open("Input_graph_final", "rb"))

pr = page_rank(G, d=.85)
print("PageRank", pr)
get_top_images(pr, K)
