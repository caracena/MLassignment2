import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score

class HandwritingPredictor:

    def getNumberFromList(self,list):
        number = np.where(list == 1)[0][0]
        return number

    def displayNumbers(self):

        # create matrix for image
        for j in range(10):
            index = self.y_labels.index(j)
            img = np.reshape(self.X.ix[index, :].values, (16, 16))
            img = np.concatenate((np.zeros((16, 2)), img), axis=1)
            count = 0
            for i in range(index+1,self.X.shape[1]):
                temp = np.reshape(self.X.ix[i, :].values, (16, 16))
                img = np.concatenate((img, temp), axis=1)
                img = np.concatenate((img, np.zeros((16,2))), axis=1)
                count += 1
                if count == 10:
                    if j == 0:
                        image = img.copy()
                        image = np.concatenate((np.zeros((2, image.shape[1])),image), axis=0)
                    else:
                        image = np.concatenate((image, img), axis=0)
                        image = np.concatenate((image, np.zeros((2,image.shape[1]))), axis=0)
                    break


        # Plot image
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
        plt.show()

    def loadFiles(self,name):
        # Loading Files
        allData = pd.read_csv(name,header=None,sep=" ")
        self.X = allData.ix[:,:255]
        self.y = allData.ix[:,256:265]
        self.y_labels = [(row[row == 1].index[0] - 256) for index, row in self.y.iterrows()]

    def predict(self,classifier,folds=10):
        #Run a cross validation on the give classifier.
        scores = cross_val_score(classifier, self.X, self.y_labels, cv=folds, scoring="accuracy");
        return scores.mean()

