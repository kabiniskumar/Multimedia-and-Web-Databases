from pymongo import MongoClient
import numpy
import tensorly as t

connection = MongoClient('localhost', 27017)
db = connection.MWDB_devset


#
# def get_mongodb_collection(mongodb_host, database, collection):
#     client = MongoClient(mongodb_host)
#     db = client.get_database(database)
#     return db.get_collection(collection)


# def get_all_objects(quantity):
#     obj = get_mongodb_collection('localhost', 'MWDBTest', quantity)
#     objlist = obj.find()
#     return objlist

def get_terms(item):
    termarray = []
    for text in item["details"]:
        termarray.append(text['term'])
    return termarray


# Connecting to Mongo DB
client = MongoClient()

# Get the database
db = client.MWDB_devset

userlist = list(db.users.find())
locationlist = list(db.locations.find())
imagelist = list(db.images.find())

userterms = dict()
locationterms = dict()
imageterms = dict()

userids = []
locationids = []
imageids = []

for location in locationlist:
    locationterms[location['number']] = get_terms(location)
    locationids.append(location['number'])
for image in imagelist:
    imageterms[image['imageID']] = get_terms(image)
    imageids.append(image['imageID'])

k = len(imageids) / 15
imageids1 = imageids[0:int(k)]
imageids2 = imageids[int(k):(int(k) * 2)]
imageids3 = imageids[(int(k) * 2):(int(k) * 3)]
imageids4 = imageids[(int(k) * 3):(int(k) * 4)]
imageids5 = imageids[(int(k) * 4):(int(k) * 5)]
imageids6 = imageids[(int(k) * 5):(int(k) * 6)]
imageids7 = imageids[(int(k) * 6):(int(k) * 7)]
imageids8 = imageids[(int(k) * 7):(int(k) * 8)]

def create_tensor(imageids):
    i = 0
    j = 0
    k = 0
    users = []
    for user in userlist:
        userterms = get_terms(user)
        locationarray = []
        j = 0
        for item in locationids:
            common_elem = list(set(userterms).intersection(locationterms[item]))
            images = []
            k = 0
            for elem in imageids:
                common_term = list(set(common_elem).intersection(imageterms[elem]))
                images.append(len(common_term))
                k += 1
            locationarray.append(images)
            j += 1
            users.append(locationarray)
        i += 1
    return users[0:700]


a = create_tensor(imageids1)
b = create_tensor(imageids2)
c = create_tensor(imageids3)
d = create_tensor(imageids4)
e = create_tensor(imageids5)
f = create_tensor(imageids6)
g = create_tensor(imageids7)
h = create_tensor(imageids8)


# Decompose tensor using CP-ALS


def write_to_file(i, list):
    path = "test" + str(i) + ".npy"
    numpy.save(path, list)

tensor1 = t.tensor(a)
write_to_file(1, tensor1)
del tensor1
tensor2 = t.tensor(b)
write_to_file(2, tensor2)
del tensor2
tensor3 = t.tensor(c)
write_to_file(3, tensor3)
del tensor3
tensor4 = t.tensor(d)
write_to_file(4, tensor4)
del tensor4
tensor5 = t.tensor(e)
write_to_file(5, tensor5)
del tensor5
tensor6 = t.tensor(f)
write_to_file(6, tensor6)
del tensor6
tensor7 = t.tensor(g)
write_to_file(7, tensor7)
del tensor7
tensor8 = t.tensor(h)
write_to_file(8, tensor8)
del tensor8
