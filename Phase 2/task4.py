import csv
import xml.etree.ElementTree
import os
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import LatentDirichletAllocation

XMLMappingFile=input("Enter XML file path for location id to name mapping")
CSV_folder= input("Enter the directory path for CSV files ")
query= input("Enter query(Location ID, Visual Descriptor Model, K)")
DimensionalityRedAlgo= input("Select and Enter Dimensionality Reduction Algorithm(PCA/ SVD/ LDA)")


allReducedLocs= []
QfileIndex=0


queryFields=query.split(" ")
QLoc=queryFields[0]
QModel=queryFields[1]
KValue=int(queryFields[2])


e = xml.etree.ElementTree.parse(XMLMappingFile).getroot()
searchQ="topic[number='"+ QLoc+"']"
item=e.find(searchQ)
name=item[1].text
QFileName= name+" "+QModel+".csv"


def WriteFileToMatrix(filename):
    ListOfLists=[]
    with open(CSV_folder+"\\"+filename,"r",) as file:
        reader = csv.reader(file)
        for row in reader:
            TemList=[]
            for element in row:
               TemList.append(float(element))
            TemList.pop(0)
            ListOfLists.append(TemList)
    return ListOfLists


def GetAllFilesByModel(modelName):
    filenames=[]
    for filename in os.listdir(CSV_folder):
        if(filename.endswith(modelName+".csv")):
            filenames.append(filename)
    return filenames


def GetAllMatriciesOfModel(model):
    allMatricies=[]
    allFiles=GetAllFilesByModel(model)
    i=0
    for item in allFiles:
        allMatricies.append(WriteFileToMatrix(item))
        if(item==QFileName):
            QfileIndex=i
        i=i+1
    return allMatricies


def ApplyPCA(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    pca=decomposition.PCA(n_components=KValue)
    pca.fit(train_data)     
    
    for loc in test_data:
        std_loc= scaler.transform(loc)
        allReducedLocs.append(pca.transform(std_loc))

    S= np.diag(pca.singular_values_)
    VT=pca.components_
    latentSemantics= (np.dot(S,VT)).transpose()

    return latentSemantics

def ApplySVD(train_data, test_data):
    svd= TruncatedSVD(n_components=KValue)
    svd.fit(train_data) 

    for loc in test_data:
        allReducedLocs.append(svd.transform(loc))
    
    S= np.diag(svd.singular_values_)
    VT=svd.components_
    latentSemantics= (np.dot(S,VT)).transpose()

    return latentSemantics


def ApplyLDA(train_data, test_data):
    lda = LatentDirichletAllocation(n_components=KValue)
    lda.fit(train_data)     
    
    for loc in test_data:
        allReducedLocs.append(lda.transform(loc))
    
    return lda.components_


qMatrix= WriteFileToMatrix(QFileName)
allMatricies= GetAllMatriciesOfModel(QModel)

if(DimensionalityRedAlgo=="PCA"):
    LatentSematics = ApplyPCA(qMatrix, allMatricies)

if(DimensionalityRedAlgo=="SVD"):
    LatentSematics = ApplySVD(qMatrix, allMatricies)

if(DimensionalityRedAlgo=="LDA"):
    LatentSematics = ApplyLDA(qMatrix, allMatricies)

ReducedQueryLoc=allReducedLocs[QfileIndex]


print("Latent semantics:")
print("-----------------------------------------------------")
print(LatentSematics)
print("dimensions: ")
print(LatentSematics.shape)
print("------------------------------------------------------")



loclocM= None
i=0
first=True
for loc in allReducedLocs:
    if(i!=QfileIndex):
        distMatrix=euclidean_distances(ReducedQueryLoc,loc)
        minDist= distMatrix.min(1)
        if(first):
            loclocM=minDist
            first=False
        else:
            loclocM=np.column_stack((loclocM, minDist))
    i=i+1

minDistIndex= np.argmin(loclocM,axis=1)

length=len(allReducedLocs)
similarityMatrix=np.zeros(length)
for item in minDistIndex:
    if(item<QfileIndex):
        similarityMatrix[item]=similarityMatrix[item]+1
    else:
        similarityMatrix[item+1]=similarityMatrix[item+1]+1
    

x=np.argsort(similarityMatrix)

print("TOP 5 SIMILAR LOCATIONS TO "+ QFileName.split(" ")[0])
allNames=GetAllFilesByModel(QModel)
i=1
while(i<=5):
    print("------------------------")
    print(allNames[x[30-i]].split(" ")[0])
    print("score:  " )
    print(similarityMatrix[x[30-i]])
    i=i+1









