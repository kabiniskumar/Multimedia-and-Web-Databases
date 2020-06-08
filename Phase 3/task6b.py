import csv
import xml.etree.ElementTree
import os
from collections import defaultdict

import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import LatentDirichletAllocation

from flask import Flask, render_template

app = Flask(__name__)

labels_images = defaultdict(list)
@app.route("/")
def visualise_top_images():
    return render_template('task6a.html', labels_images=labels_images)


###########################################################################
##GET INPUT
CSV_folder= input("Enter the directory path for CSV files ")
labelsFile=input("enter image-labels file path")
global_image_ids=[]
LabelDict={}
Labelled_ImagesIds=[]
Labels=[]
#Extract labelled images data from input file
def ExtractData():
    with open(labelsFile) as f:
         Qfile = f.read().splitlines()
    for line in Qfile:
        arr=line.split(" ")
        id=int(arr[0])
        n=len(arr)-1
        label=arr[n]
        LabelDict[id]=label
        Labelled_ImagesIds.append(id)
        if label not in Labels:
            Labels.append(label)


##Create Matrix from file
def WriteFileToMatrix(filename, id_append):
    global global_image_ids
    ListOfLists=[]
    with open(CSV_folder+"\\"+filename,"r",) as file:
        reader = csv.reader(file)
        for row in reader:
            TemList=[]
            for element in row:
               TemList.append(float(element))
            if id_append:
                 global_image_ids.append(int(TemList.pop(0)))
            ListOfLists.append(TemList)
    return ListOfLists

##Get all files names
def GetAllFileNames():
    filenames=[]
    for filename in os.listdir(CSV_folder):
        filenames.append(filename)
    return filenames

##Get all matricies of a model
def GetMatrixWithAllFeatures(i):
    Obj_feature_Matrix=None
    allFiles=GetAllFileNames()
    count=0
    first=True
    while(count<10):
         if(first):
             Obj_feature_Matrix=WriteFileToMatrix(allFiles[count+i],True)
             first=False
         else:
             Obj_feature_Matrix= np.hstack((Obj_feature_Matrix,WriteFileToMatrix(allFiles[count+i], False)))
         count=count+1
    return Obj_feature_Matrix

def GetMatrix():
    Matrix=[]
    first=True
    n=len(GetAllFileNames())
    i=0
    while(i<n):
         if(first):
             Matrix=GetMatrixWithAllFeatures(i)
             first=False
         else:
             Matrix=np.concatenate((Matrix,GetMatrixWithAllFeatures(i)))
         i=i+10
    return Matrix
#################################################################################
m=GetMatrix()
object_feature_matrix= minmax_scale(m, feature_range = (0,5), axis = 1)

ExtractData()

# Labelled_ImagesIds=[3298433827,299114458,948633075,4815295122,5898734700,4027646409,1806444675,4501766904,6669397377,3630226176,3779303606,4017014699]
# Labels=["fort","sculpture"]
# LabelDict={}
# for i in Labelled_ImagesIds:
#      LabelDict[i]="sculpture"
# LabelDict[3298433827]="fort"
# LabelDict[4027646409]="fort"
# LabelDict[1806444675]="fort"
# LabelDict[4501766904]="fort"


n=len(Labelled_ImagesIds)
Labelled_ImageFeatureMatrix=[]
AllOtherImages=[]
AllOtherImageIds=[]
for id in global_image_ids:
    AllOtherImageIds.append(id)
for item in object_feature_matrix:
    AllOtherImages.append(item)
for i in Labelled_ImagesIds:
    z= global_image_ids.index(i)
    AllOtherImageIds.pop(z)
    Labelled_ImageFeatureMatrix.append(object_feature_matrix[z])
    AllOtherImages.pop(z)


def createGraph(img):
    cc=Labelled_ImageFeatureMatrix.copy()
    cc.append(img)
    distMtrix= euclidean_distances(cc,cc)
    graph=[]
    j=9
    for img in distMtrix:
         imggraph=[0]* (len(Labelled_ImagesIds)+1)
         sortedindexes=np.argsort(img)
         count=0
         while(count<j):
             imggraph[sortedindexes[count]]=1
             count=count+1
         graph.append(imggraph)
    return graph


def pageRank(err, alpha,graph):
    n=len(Labelled_ImagesIds)+1
    M=np.array(graph)
    s = M.sum(axis=0)
    s[s == 0] = 1
    M=M/s
    rNext=np.ones((n,1))
    rNext=rNext/n
    r=np.zeros((n,1))
  
    teleport=np.zeros((n,n))
    teleport[n-1]=np.ones(n)
    
    A= alpha*M+(1-alpha)*teleport

    while(np.sum(np.abs(r - rNext))>err):
        r=rNext
        rNext=np.dot(A,r)
    return rNext

AllImagesLabelDict={}
def ApplyLabel():
    globalCount=0
    for img in AllOtherImages:
        g=createGraph(img)
        scores=pageRank(.001,0.85,g)
        maxscoreindex=np.argsort(scores, axis=0)
        qindex=len(scores)-1
        mt=np.transpose(maxscoreindex)[0]
        count=0
        labelHash={}
        for label in Labels:
            labelHash[label]=0
        while(count<2):
            if(mt[count]!=qindex):
                 ImgId=Labelled_ImagesIds[mt[count]]
                 label=LabelDict[ImgId]
                 labelHash[label]=labelHash[label]+1
            count=count+1
        maxvalue= max(labelHash,key=labelHash.get)
        AllImagesLabelDict[AllOtherImageIds[globalCount]]=maxvalue
        globalCount=globalCount+1

# def ApplyLabel():
#     globalCount=0
#     for img in AllOtherImages:
#         g=createGraph(img)
#         scores=pageRank(.001,0.85,g)
#         maxscoreindex=np.argsort(scores, axis=0)
#         top=maxscoreindex[0][0]
#         qindex=len(scores)-1
#         if top==qindex:
#              top=maxscoreindex[1][0]
#         LabelImgId=Labelled_ImagesIds[top]
#         value=LabelDict[LabelImgId]
        
#         AllImagesLabelDict[AllOtherImageIds[globalCount]]=value
#         globalCount=globalCount+1
ApplyLabel()  
for item in AllImagesLabelDict:
    key=item
    value=AllImagesLabelDict[key]
    # print(f"{key} : { value}" )


for key, value in AllImagesLabelDict.items():
    # key = [str(s) + '.jpg' for s in key]
    key = str(key) + '.jpg'
    if key not in labels_images:
        labels_images[value].append(key)

print(labels_images)
app.run()