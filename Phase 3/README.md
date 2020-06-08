# Phase 2
This phase experiments with graph analysis, clustering, indexing and classification techniques on an image dataset.


## Task 1
Given a value k, create an image-image similarity graph

`python task1.py`

## Task 2
Given the image-image similarity graph and an integer value c, identify c clusters using two distinct algoritms(K-means and Spectral Clustering) and visualize the results.

`python task2.py` <br />
The output will be displayed on the webpage (http://127.0.0.1:5000/)

## Task 3
Given an image-image similarity graph, identif yand visualize K most dominant images using Page Rank (PR) for a user supplied K.

`python task3.py` <br />
The output will be displayed on the webpage (http://127.0.0.1:5000/)

## Task 4
Given an image-image graph and 3 user specified imageids identify and visualize K most relevant images using personalized PageRank (PPR) for a user supplied K

`python task4.py` <br />
The output will be displayed on the webpage (http://127.0.0.1:5000/)

## Task 5
Implement a Locality Sensitive Hashing (LSH) tool, for a and similarity/distance function of your choice, which takes as input (a) the number of layers, L, (b) the number of hashes per layer, k, and (c) a set of vectors as input and creates an in-memory index structure containing the given set of vectors.
Implement similar image search using this index structure and a combined visual model function of your choice (the combined visual modelmust have at least 256 dimensions): for a given image and t, visulizes the t most similar images.

#### Run the below command to create LSH_index and LSH_bucket files which store the LSH index structure.
`python task5a.py <L> <K>`

`python task5b.py <query imageID> <T>` <br /> 
The output will be displayed on the webpage (http://127.0.0.1:5000/)
Image IDs will also be displayed on the terminal along with the number of total unique candidates considered.

## Task 6
Implement a k-nearest neighbor based classification algorithm and Personalized PageRank based classification algorithm

`python 6a.py`

`python 6b.py` <br />
The output will be displayed on the webpage (http://127.0.0.1:5000/)
