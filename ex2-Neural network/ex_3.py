import sys
import numpy as np

numNeurons=128
sigmoid = lambda x: 1 / (1 + np.exp(-x))



def shuffleData(train_x,train_y):
    zip_info = list(zip(train_x, train_y))
    np.random.shuffle(zip_info)
    train_x, train_y = zip(*zip_info)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

def normalization(train_x):
    train_new=np.divide(train_x,255)
    return train_new

def softmax(z):
    z = z - np.max(z)
    zExp=np.exp(z)
    sumE=np.sum(zExp)
    new_z=np.divide(zExp,sumE)
    return new_z

def lossNLL(h):
    arrLog=np.log(h)
    arrN=np.negative(arrLog)
    loss=np.sum(arrN)
    return loss

def fprop(x, y, params):
  # Follows procedure given in notes
  w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
  z1 = np.dot(w1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(w2, h1) + b2
  # softmax
  h2 = softmax(z2)
  # loss with softmax
  loss = lossNLL(h2)
  ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret

def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    vectorY=np.zeros(shape=(10,1),dtype=int)
    # hot encoding , put 1 only in the true value of y
    index=(int)(y)
    vectorY[index,]=1
    # dertive of softmax ,becauze now we in multiclass
    dz2 = h2-vectorY  # dL/dz2
    dw2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['w2'].T,
                dz2) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    # new_x=np.reshape(x,(1,784))
    dw1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'db1': db1, 'dw1': dw1, 'db2': db2, 'dw2': dw2}


def trainAlgo(train_x,train_y):
    epochs = 13
    # initialize
    w1 = np.random.uniform(-1,1,[numNeurons ,784])*0.1
    b1 = np.random.rand(numNeurons, 1)*0.1
    w2 = np.random.uniform(-1,1,[10, numNeurons])*0.1
    b2 = np.random.rand(10, 1)*0.1
    # learning rate
    lr=0.03
    for e in range(epochs):
        train_x, train_y = shuffleData(train_x, train_y)
        # pick example
        for x, y in zip(train_x, train_y):
            sizeR=train_x.shape[1]
            x=np.reshape(x,(sizeR,1))
            params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
            # foward through network +calc the loss
            fprop_cache=fprop(x,y,params)
            # compute the gradients
            bprop_cache = bprop(fprop_cache)
            #update the parameters
            db1, dw1, db2, dw2 = [bprop_cache[key] for key in ('db1', 'dw1', 'db2', 'dw2')]
            w1=w1-lr*dw1
            w2=w2-lr*dw2
            b1=b1-lr*db1
            b2=b2-lr*db2

    return w1,w2,b1,b2

def testAlgo(w1,w2,b1,b2,test_x):
    size=test_x.shape[0]
    test_y = size* [None]
    index=0
    for x in test_x:
        sizeR = test_x.shape[1]
        x = np.reshape(x, (sizeR, 1))
        params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
        fprop_cache=fprop(x,None,params)
        h2=fprop_cache['h2']
        test_y[index]=np.argmax(h2)
        index=index+1
    return test_y


# main
# read the files
trainXFile = sys.argv[1]
trainYFile=sys.argv[2]
testXFile=sys.argv[3]
train_x=np.loadtxt(trainXFile,dtype='uint8')
train_y=np.loadtxt(trainYFile)
test_x=np.loadtxt(testXFile)
train_xN=normalization(train_x)
w1,w2,b1,b2=trainAlgo(train_xN,train_y)
test_xN=normalization(test_x)
test_y=testAlgo(w1,w2,b1,b2,test_xN)
# train_x=normalization(train_x)
f = open("test_y", "w")
for y in test_y:
    f.write(str(y)+"\n")
f.close()
