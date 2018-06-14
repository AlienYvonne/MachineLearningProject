import numpy as np
from PIL import Image

from sklearn.mixture.gaussian_mixture import GaussianMixture
from sklearn.decomposition import PCA

def importTrainData():
    data_num = 60000  # The number of figures
    fig_w = 45  # width of each figure
    data = np.fromfile("mnist_train_data", dtype=np.uint8)
    label = np.fromfile("mnist_train_label", dtype=np.uint8)

    # reshape the matrix
    data = data.reshape(data_num, fig_w * fig_w)

    return data, label

def importTestData():
    data_num = 10000  # The number of figures
    fig_w = 45  # width of each figure
    data = np.fromfile("mnist_test_data", dtype=np.uint8)
    label = np.fromfile("mnist_test_label", dtype=np.uint8)

    # reshape the matrix
    data = data.reshape(data_num, fig_w * fig_w)

    return data, label

def main():
    #Import data from file
    trainData, trainLabel = importTrainData()
    testData, testLabel = importTestData()


    #PCA procedure

     #For train data
    rawlowDivTrainData = []
    pca = PCA(n_components=537)
    #pca = PCA(n_components=537, whiten="True")

    rawlowDivTrainData = pca.fit_transform(trainData)

     #For test data which sum up to 10000
    lowDivTestData = pca.fit_transform(testData)

     #Classify manually, divide the data into 10 parts by labels
    numTrainData = [[], [], [], [], [], [], [], [], [], []]
    for i in range(60000):
        numTrainData[trainLabel[i]].append(rawlowDivTrainData[i])

     #Train Gmm for every mode/Each mode represent a number.
    myGmms = []
    #GMM procedure
    for num in range(10):
        print("GMM", num)
        gmmClassifier = GaussianMixture(n_components=25, init_params='kmeans', max_iter=1000)
        gmmClassifier.fit(numTrainData[num])
        myGmms.append(gmmClassifier)

    #Validate
    # We need to calculate all the total probability under each GMM, and choose the biggest one.
    correctCount = 0
    allProb = []
    for num in range(10):
        allProb.append(np.array(myGmms[num].score_samples(lowDivTestData)))

    allProb = np.array(allProb)
        #If the result seem strange, print the predict label

    for i in range(10000):
        predictLabel = 0
        max = np.max(allProb[:, i])
        for num in range(10):
            if max == allProb[num, i]:
                predictLabel = num
        if predictLabel == testLabel[i]:
            correctCount += 1

    #Output
    print("The correct rate is:", correctCount/10000)

main()
