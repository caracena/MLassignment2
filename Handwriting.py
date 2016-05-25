import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score

class HandwritingPredictor:

    def getNumberFromList(self,list):
        number = np.where(list == 1)[0][0]
        return number;

    def displayNumber(self,numberArray,numberArr):
        # Transforms array of 10 numbers into actual number
        actualNumber = self.getNumberFromList(numberArr)
        # Shows actual number
        print(actualNumber)

        # Building image array with 16x16 pixels
        #.
        i = 16
        k = 0
        img = np.array(numberArray[:16])
        #Each 16 elements corresponds to one line of the 16 x 16 matrix
        while i <= len(numberArray):

            temp = np.array(numberArray[k:i])
            img = np.vstack((img,temp))

            k = i
            i += 16

        # Plot image
        plt.imshow(img,cmap=plt.cm.gray_r,interpolation="nearest")
        plt.show()

    def loadFiles(self,name):
        # Loading Files
        allData = pd.read_csv(name,header=None,sep=" ")
        X = allData.ix[:,:255]
        y = allData.ix[:,256:265]
        y_labels = [(row[row == 1].index[0] - 256) for index, row in y.iterrows()]
        return X, y_labels, y

    def predict(self,classifier,X,y,folds=10):
        #Run a cross validation on the give classifier.
        scores = cross_val_score(classifier, X, y, cv=folds, scoring="accuracy");
        return scores.mean()

