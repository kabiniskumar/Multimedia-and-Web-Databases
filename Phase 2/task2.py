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
import xml.etree.ElementTree as ET
import scipy
import logging
logging.getLogger("gensim").setLevel(logging.CRITICAL)
model="DF"
#k=int(input("enter k value"))
k=10
space=input("Enter vector space (UT,IT or LT)")
if space=="UT":
    location=input("Enter directory location with file path:(with double slashes)")
    extractdata= extract_info.DataExtractor()
    df=extractdata.openandextractdata(location)
    dim=input("Enter model (SVD,PCA or LDA)")
    users = df["Userid"].unique()
    entities=users
    dfstring="Userid"
    count=0
    userid=input("Enter Userid")
    #userid='23260223@N04'
    for i in users:
        if(i==userid):
            break
        count+=1

elif space =="IT":
    location=input("Enter directory location with file path:(with double slashes)")
    extractdata= extractinfoimage.DataExtractor()
    df=extractdata.openandextractdataimage(location)
    dim=input("Enter model (SVD,PCA or LDA)")
    images = df["Imageid"].unique()
    entities=images
    dfstring="Imageid"
    count=0
    imageid=input("Enter Imageid")
#    imageid='135405610'

    for i in images:
        if(i==imageid):
            break
        count+=1
    print("Running...count=%d"%(count))



else:
    location=input("Enter directory location with file path:(with double slashes)")
    extractdata= extractinfolocation.DataExtractor()
    df=extractdata.openandextractdatalocation(location)
    dim=input("Enter model (SVD,PCA or LDA)")
    locations = df["Location"].unique()
    entities=locations
    dfstring="Location"
    locationnumber=int(input("enter location number"))
    count=locationnumber


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
    V=np.transpose(V)
    ncrossv=np.matmul(matrix,V)

    ncrossvtrans=np.transpose(ncrossv)
    similaritymatrixSVD=np.matmul(ncrossv,ncrossvtrans)
    similaritymatrixSVD[count][count]=-1
    similarentities= nlargest(5,enumerate(similaritymatrixSVD[count]),key=lambda  x:x[1])
    print("Most similar users and matching scores are:")
    j=1
    for i in range(5):
        print("%s %f"%(entities[similarentities[i][0]],(similarentities[i][1])/math.log1p(j)))
        j+=1
    print("####################################################SVDEND#############################################")
if dim=='PCA':
    print("###################################################PCA#################################################")
    covariance = np.cov(matrix, rowvar=False)
    U, s, Vh = scipy.sparse.linalg.svds(covariance,k)

    Vh=np.transpose(Vh)
    ncrossk=np.matmul(matrix,Vh)
    ncrossktrans=np.transpose(ncrossk)
    similaritymatrixPCA=np.matmul(ncrossk,ncrossktrans)
    similaritymatrixPCA[count][count]=-1
    similarusers= nlargest(5,enumerate(similaritymatrixPCA[count]),key=lambda  x:x[1])
    print("Most similar users and matching scores are:")
    j=1
    for i in range(5):
        print("%s %f"%(entities[similarusers[i][0]],(similarusers[i][1])/math.log10(j+1)))
        j+=1
    print("####################################################################PCA#########################################")
##################################################LDA PART###################################################
if dim=='LDA':
    print("#########################################################LDA##############################################################")
    new_df=pd.DataFrame(columns=['Userid','Term'])
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

    u_matrix = np.zeros(shape=(len(entities), 10))
    for i in range(0, len(U)):
        doc = U[i]
        for j in range(0, len(doc)):
            (topic_no, prob) = doc[j]
            u_matrix[i, topic_no] = prob

    latent_entity_matrix = u_matrix.transpose()
    entity_entity_matrix = np.dot(u_matrix, latent_entity_matrix)
    entity_entity_matrix[count][count]=-1
    ksimilarentities=nlargest(5, enumerate(entity_entity_matrix[count]), key=lambda x:x[1])
    print("Most similar users are:")
    for i in range(5):
        print("%s %f"%(entities[ksimilarentities[i][0]],ksimilarentities[i][1]))
