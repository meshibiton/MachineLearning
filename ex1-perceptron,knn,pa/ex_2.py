import sys
import numpy as np

# to convert nominal col
def conv(wineT):
    if(wineT==b'R'):
        return 1
    return 0

# read the file and convert the 11 column-check
def readFileToMatrix(file1,file2,file3):
    train_x = np.loadtxt(file1,delimiter=",",converters={11:conv})
    train_y=np.loadtxt(file2,dtype=int)
    test_x= np.loadtxt(file3,delimiter=",",converters={11:conv})
    return train_x,train_y,test_x

def nirmolData(matrix,minFeature,maxFeature):
    matrixNorm=np.zeros_like(matrix)
    index=0
    for vector in matrix:
        # we calculate
        # each vector ,by normalize all the feature at one time
        v_new=(np.subtract(vector, minFeature))/(np.subtract(maxFeature, minFeature))
        # add the row to the matrix
        matrixNorm[index] = v_new
        index = index + 1
    return matrixNorm

def findMaxAndMin(train_x):
    numF=train_x.shape[1]
    minFeature=numF*[None]
    maxFeature=numF*[None]
    index=0
    # go throung each colum and find the max and min of each feature
    for col in range(train_x.shape[1]):
        max=np.amax(train_x[:, col])
        min=np.amin(train_x[:, col])
        minFeature[index]=min
        maxFeature[index]=max
        index=index+1
    return maxFeature,minFeature


def KnnAlgo(test_x,train_x,train_y,k):
    sizeTest=test_x.shape[0]
    test_y=np.zeros(shape= (sizeTest,), dtype=int)
    index=0
    for point in test_x:
       #  find the distance between the point to the train set
       sub_point = np.subtract(point, train_x)
       distanceArray=np.linalg.norm(sub_point,axis=1)
       # take the k nearest points - save the index of
       distanceIndex = distanceArray.argsort()[:k]
       # take the lable of each index into array
       Kclasses = np.copy(train_y[distanceIndex,])
       occurances = np.bincount(Kclasses)
       # take the most freq class and the smallest if there are 2
       max = np.argmax(occurances)
       # lable the test point to the chosen class
       test_y[index]=max
       index=index+1
    return test_y

def shuffleData(train_x,train_y):
    zip_info = list(zip(train_x, train_y))
    np.random.shuffle(zip_info)
    train_x, train_y = zip(*zip_info)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

# find the y_hat in pa
def findY(w,x,y):
    arr=np.dot(w, x)
    maxIndex=np.argmax(arr)
    if(maxIndex==y):
        minIndex = np.argmin(arr)
        arr[maxIndex]=arr[minIndex]-1
        return np.argmax(arr)
    return maxIndex

def paAlgo(train_x,train_y):
    # we have 3 classes and 13 features(include bias)
    w = np.zeros(shape=(3, 13))
    w.fill(1)
    train_x, train_y = shuffleData(train_x, train_y)
    for x, y in zip(train_x, train_y):
        # predict
        #  we want the biggest except to him
        y_hat = findY(w,x,y)
        loss=max(0,1-(np.dot(w[y, :],x))+(np.dot(w[y_hat,:],x)))
        # update
        if loss is not 0:
            tauDeno=2*((np.linalg.norm(x))**2)
            if tauDeno is not 0:
                tau=loss/tauDeno
                w[y, :] = w[y, :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x

    return w

def perAlgo(train_x,train_y,eta):
    # we have 3 classes and 13 features(include bias)
    w = np.zeros(shape=(3, 13))
    w.fill(1)
    epochs = 12
    for e in range(epochs):
        train_x, train_y = shuffleData(train_x, train_y)
        for x, y in zip(train_x, train_y):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            # update
            if y != y_hat:
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
    return w

def testPerAndPa(w,test_x):
    # now we find the test predicted y ,according to w we founded
    sizeTest = test_x.shape[0]
    test_y = np.zeros(shape=(sizeTest,), dtype=int)
    index = 0
    for x in test_x:
        test_y[index] = np.argmax(np.dot(w, x))
        index = index + 1
    return test_y

def calcAccuracy(predictid_y,test_y,numTest):
    countCorrect=np.sum(predictid_y== test_y)
    accuaracy= countCorrect/numTest
    return accuaracy

def addBias(matrix):
    num=matrix.shape[0]
    bias = np.zeros(shape=(num, 1))
    bias.fill(1)
    new_matrix = np.append(matrix, bias, axis=1)
    return new_matrix



# read the files
trainXFile = sys.argv[1]
trainYFile=sys.argv[2]
testXFile=sys.argv[3]
train_x,train_y,test_x=readFileToMatrix(trainXFile,trainYFile,testXFile)
# find min and max of each feature
maxF,minF=findMaxAndMin(train_x)
# nirmol
train_x=nirmolData(train_x,minF,maxF)
test_x=nirmolData(test_x,minF,maxF)
# bias just in pa and perc ,not knn
train_xB=addBias(train_x)
test_xB=addBias(test_x)
# test the points in knn algo with k=7 as in the report.
knn_yhatArr=KnnAlgo(test_x,train_x,train_y,7)
# train -perceptron algo with eta=0.001 and epoch-12
wPe=perAlgo(train_xB,train_y,0.001)
# predict y
per_yhatArr=testPerAndPa(wPe,test_xB)
# train pa algo
wPa=paAlgo(train_xB,train_y)
# predict y.
pa_yhatArr=testPerAndPa(wPa,test_xB)
index=0
for x in test_x:
      print(f"knn: {knn_yhatArr[index]}, perceptron: {per_yhatArr[index]}, pa: {pa_yhatArr[index]}")
      index=index+1


