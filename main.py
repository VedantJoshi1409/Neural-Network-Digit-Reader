import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

data = np.array(pd.read_csv("digit-recognizer/train.csv"))
# testData = np.array(pd.read_csv("digit-recognizer/test.csv")).T
# testData does not have any labels :sob:
dataAmount, imageDataLength = data.shape

testData = data[0:1000].T
trainData = data[1000:dataAmount].T
# transpose data so that each column is a label + img instead of each row

dataUsed = trainData  # for easy switching between test and train
iterations = 500
saveData = False
loadParam = True  # use random weights and biases or not

labels = dataUsed[0]
pixels = dataUsed[1:imageDataLength]
pixels = pixels / 255


# squish values into something between 0 and 1


def initParams():
    W1 = np.random.rand(10, 784) - 0.5  # values range from -0.5 to 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def saveParams(W1, b1, W2, b2):
    np.save("Params/W1.npy", W1)
    np.save("Params/b1.npy", b1)
    np.save("Params/W2.npy", W2)
    np.save("Params/b2.npy", b2)


def loadParams():
    W1 = np.load("Params/W1.npy")
    b1 = np.load("Params/b1.npy")
    W2 = np.load("Params/W2.npy")
    b2 = np.load("Params/b2.npy")
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def softMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    # np.exp(Z) calc
    return A


def forwardProp(W1, b1, W2, b2, X):
    # X is the input layer values
    Z1 = W1.dot(X) + b1
    # Z1 is the unactivated first layer
    # W1.dot(X) performs matrix multiplication of W1 * X. Result is a 10x41000 (data amount) array
    A1 = ReLU(Z1)
    # A1 is activated first layer

    Z2 = W2.dot(A1) + b2
    A2 = softMax(Z2)

    return Z1, A1, Z2, A2


def oneHotValues(Y):
    # Values to compare output layer with to find error amount
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY


def derivReLU(Z):
    return Z > 0
    # true converts to 1 and false converts to 0


def backProp(Z1, A1, W1, Z2, A2, W2, X, Y):
    m = Y.size
    # amount of data

    dZ2 = A2 - oneHotLabels
    # amount of error in output layer

    dW2 = 1 / m * dZ2.dot(A1.T)
    # used to see how much of the error is caused by W2

    db2 = 1 / m * np.sum(dZ2)
    # used to see how much of the error is caused by b2

    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)  # undoes the weight multiplication and ReLU Function
    dW1 = 1 / m * dZ1.dot(X.T)  # same as before
    db1 = 1 / m * np.sum(dZ1)
    # idk how exactly the calculations work out though

    return dW1, db1, dW2, db2


def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # subtract the error times alpha from the weight to try to get closer to the perfect value
    # not subtracting full error since that would work for the 1 piece of data but will mess up all over results
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def getPredictions(A2):
    return np.argmax(A2, 0)


def getAccuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def getBatches(pixels, labels):
    pixelsT = pixels.T
    labelsT = labels.T
    labelBatches = []
    pixelBatches = []
    batchSize = 100
    for i in range(0, len(labels), batchSize):
        batch = labelsT[i:i + batchSize].T
        labelBatches.append(batch)
        batch = pixelsT[i:i + batchSize].T
        pixelBatches.append(batch)
    return pixelBatches, labelBatches


def gradientDescent(X, Y, iterations, alpha):
    if loadParam:
        W1, b1, W2, b2 = loadParams()
    else:
        W1, b1, W2, b2 = initParams()

    for i in range(iterations):
        global oneHotLabels
        oneHotLabels = oneHotValues(Y)
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        # find what the NN predicts
        dW1, db1, dW2, db2 = backProp(Z1, A1, W2, Z2, A2, W2, X, Y)
        # find error amount from prediction
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # tune to reduce error
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", getAccuracy(getPredictions(A2), Y))

    return W1, b1, W2, b2


def gradientDescentBatches(X, Y, iterations, alpha):
    if loadParam:
        W1, b1, W2, b2 = loadParams()
    else:
        W1, b1, W2, b2 = initParams()
    XBatch, YBatch = getBatches(X, Y)

    for i in range(iterations):
        for j in range(len(XBatch)):
            global oneHotLabels
            oneHotLabels = oneHotValues(YBatch[j])
            Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, XBatch[j])
            # find what the NN predicts
            dW1, db1, dW2, db2 = backProp(Z1, A1, W2, Z2, A2, W2, XBatch[j], YBatch[j])
            # find error amount from prediction
            W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            # tune to reduce error
            if i % 10 == 0 and j == 0:
                oneHotLabels = oneHotValues(Y)
                Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
                # find what the NN predicts
                dW1, db1, dW2, db2 = backProp(Z1, A1, W2, Z2, A2, W2, X, Y)
                # find error amount from prediction
                W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

                print("Iteration: ", i)
                print("Accuracy: ", getAccuracy(getPredictions(A2), Y))

    return W1, b1, W2, b2


def predictImg():
    image = cv2.imread('ImageDropbox/digit.png')[:, :, 0]
    image = np.invert(np.array([image]))
    imageArray = np.array(image)
    X = imageArray.flatten().reshape(-1, 1)
    X = X / 255

    W1, b1, W2, b2 = loadParams()
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    print("The number is probably a {}".format(getPredictions(A2)))
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.show()


# predictImg()
oneHotLabels = 0  # just declare variable
W1, b1, W2, b2 = gradientDescent(pixels, labels, iterations, 0.1)
if saveData and loadParam:  # if random params, shouldn't save them :skull:
    saveParams(W1, b1, W2, b2)
