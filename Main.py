from Handwriting import HandwritingPredictor
from sklearn import svm
# Create a classifier and feed into the predict function. Cross validation will be automatically used
def main():
    predictor = HandwritingPredictor()
    X, y_labels = predictor.loadFiles("semeion.data")
    # X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, y_labels, test_size=0.33, random_state=42)

    clf = svm.SVC(gamma=0.01, C=100)
    print(predictor.predict(clf, X, y_labels))

if __name__ == "__main__":
    main()