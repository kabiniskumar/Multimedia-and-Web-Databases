import csv
import xml.etree.ElementTree
import os
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import euclidean_distances
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
KValue=input("Enter K value for KNN")
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

ExtractData()
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
distMtrix= euclidean_distances(AllOtherImages,Labelled_ImageFeatureMatrix)

AllImagesLabelDict={}
def ApplyLabel(kValue):
    globalCount=0
    for dist in distMtrix:
        
        labelHash={}
        for label in Labels:
            labelHash[label]=0
        sortedIndexes=np.argsort(dist)
        count=0
        while(count<kValue):
            ImgId=Labelled_ImagesIds[sortedIndexes[count]]
            label=LabelDict[ImgId]
            labelHash[label]=labelHash[label]+1
            count=count+1
        maxvalue= max(labelHash,key=labelHash.get)
        AllImagesLabelDict[AllOtherImageIds[globalCount]]=maxvalue
        globalCount=globalCount+1

ApplyLabel(int(KValue))
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

