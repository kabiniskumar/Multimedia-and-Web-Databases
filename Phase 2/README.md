# Phase 2

This phase focuses on latent semantics and identifying relationships between objects and features in a reduced dimensional space.
The project experiments with techniques like PCA, LDA, SVD and Tensors(CP-Decomposition).

## Task 1
Identify latent semantics with respect to objects and represent them in a reduced dimensional space.
The program lets the user choose among 1) User-term vector space 2) Image-term vector space and 3)Location-term vector space.
Then, given an integer k, identify and report top-k latent semantics/topics in the corresponding term space using PCA/SVD/LDA.

`python <file-path> task1.py`

## Task 2
Given a userID, imageID or locationID, identify the most related 5 userIDs, imageIDs, locationIDs.

`python <file-path> task2.py`

## Task 3
Identify 5 most similar images and locations given an image ID, a visual descriptor model(CM, CM3x3, CN, CN3x3, CSD, GLRLM, GLRLM3x3, HOG, LBP, LBP3x3 ) and a dimensionality reduction technique.

`python task3.py <k> <visual-model> <reduction-algorithm> <image-ID>`

## Task 4
Identify 5 most similar locations given a location ID, a visual descriptor model and a dimensionality reduction technique.

`python task4.py`

## Task 5
Identify 5 most similar locations given a location ID and a dimensionality reduction technique.

`python task5.py`

## Task 6
Given a value k, create a location-location similarity matrix and report top-k latent semantics.

`python task6.py`

## Task 7
Given a value k, create a user-image-location tensor, perform rank-k CP decomposition and finally create k non-overlapping groups of users, images and locations based on the discovered latent semantics.

#### Run the below command to create a tensor of users, images, and locations
`python create_tensor.py`

`python task7.py`


