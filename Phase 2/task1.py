import extract_info
import extractinfolocation
import extractinfoimage
import os
import pandas as pd
import numpy as np
from heapq import nsmallest
from collections import defaultdict
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
from heapq import nlargest
import gensim
from gensim import corpora
import scipy
import logging
logging.getLogger("gensim").setLevel(logging.CRITICAL)
model="DF"
k=int(input("enter k value"))
space=input("Enter vector space (UT IT or LT)")
if space=="UT":
    location=input("Enter directory location with file path:(with double slashes)")
    extractdata= extract_info.DataExtractor()
    df=extractdata.openandextractdata(location)
    dim=input("Enter model (SVD,PCA or LDA)")
    users = df["Userid"].unique()
    entities=users
    dfstring="Userid"


elif space =="IT":
    location=input("Enter directory location with file path:(with double slashes)")
    extractdata= extractinfoimage.DataExtractor()
    df=extractdata.openandextractdataimage(location)
    dim=input("Enter model (SVD,PCA or LDA)")
    images = df["Imageid"].unique()
    entities=images
    dfstring="Imageid"
else:
    location=input("Enter directory location with file path:(with double slashes)")
    extractdata= extractinfolocation.DataExtractor()
    df=extractdata.openandextractdatalocation(location)
    dim=input("Enter model (SVD,PCA or LDA)")
    locations = df["Location"].unique()
    entities=locations
    dfstring="Location"

terms = df["Term"].unique()
columns = list(terms)
#extract dataframe and place in a matrix(User x Term) containing the model values
matrix = pd.DataFrame(columns = columns)
for entity in entities:

    vector = []
    entity_df = df[df[dfstring] == entity]
    entity_terms = list(entity_df["Term"].unique())
    for term in terms:
        if term not in entity_terms:
            vector.append(0)
        else:
            new_df = entity_df[entity_df["Term"] == term]
            vector.append(float(new_df[model].iloc[0]))
    matrix.loc[len(matrix.index)] = vector

############################################SVDPART#########################################################
if dim=='SVD':
    print("#########################################################SVD################################################################")
    #U, s, V = np.linalg.svd(matrix,full_matrices=False)
    U, s, V = scipy.sparse.linalg.svds(matrix,k)
    transmatrix=np.transpose(matrix)
    V=np.matmul(transmatrix,U)
    V=np.transpose(V)
    kuserarray=[]
    print("Top k latent semantics are:")
    for i in range(k):
        kusers=nlargest(10, enumerate(V[i]), key=lambda x:x[1])
        kuserarray.append(kusers)

    kuserarray.sort()

    for i in range(k):
        print("\nlatent semantic %d"%(i+1))
        for j in range(10):
            print("%f*%s"%(abs(kuserarray[i][j][1]),terms[kuserarray[i][j][0]]),end="")
            print("\n",end="")
#print("####################################################SVDEND#############################################")
if dim=='PCA':
    print("###################################################PCA#################################################")
    covariance = np.cov(matrix, rowvar=False)
    U, s, Vh = np.linalg.svd(covariance,full_matrices=False)
    #U, s, Vh = scipy.sparse.linalg.svds(covariance,k)
    V=np.matmul(covariance,U)
    kuserarray=[]
    print("Top k latent semantics are:")
    for i in range(k):
        kusers=nlargest(10, enumerate(V[i]), key=lambda x:x[1])
        kuserarray.append(kusers)

    kuserarray.sort()
    for i in range(k):
        print("\nLatent semantic %d"%(i+1))
        for j in range(10):
            print("%f*%s"%(abs(kuserarray[i][j][1]),terms[kuserarray[i][j][0]]),end="")
            print(" \n",end="")
# print("####################################################################PCA#########################################")
# ##################################################LDA PART###################################################
#
# print("#########################################################LDA##############################################################")
if dim=='LDA':
    print("##############################################################LDA###########################################################")
    new_df=pd.DataFrame(columns=[dfstring,'Term'])
    i=0
    listoflists=[]
    for entity in entities:
        listofterms=list(df.loc[df[dfstring]==entity,'Term'])
        listoflists.append(listofterms)

    dictionary = corpora.Dictionary(listoflists)
    corpus = [dictionary.doc2bow(text) for text in listoflists]
    lda = gensim.models.ldamodel.LdaModel(corpus, k, id2word=dictionary, passes=20)
    latent_semantics = lda.print_topics(k, len(terms))
    corpus=lda[corpus]
    U=corpus
    Vh=latent_semantics
    for latent in latent_semantics:
        print(latent)



