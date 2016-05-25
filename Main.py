from Handwriting import HandwritingPredictor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Create a classifier and feed into the predict function. Cross validation will be automatically used
def main():
    predictor = HandwritingPredictor()
    X, y_labels, y = predictor.loadFiles("semeion.data")

    models = [svm.SVC(gamma=0.01, C=100),
              RandomForestClassifier(n_estimators=500, max_features='auto', n_jobs = -1),
              AdaBoostClassifier(n_estimators=500),
              DecisionTreeClassifier(max_features ='auto' ,max_depth=5),
              MultinomialNB(),
              GaussianNB(),
              LogisticRegression(solver='lbfgs', multi_class = 'ovr', max_iter = 50 ,n_jobs = -1),
              KNeighborsClassifier(3, n_jobs = -1),
              LinearDiscriminantAnalysis(),
              QuadraticDiscriminantAnalysis()]

    for model in models:
        print('accuracy for model {}: {}'.format(model.__str__,predictor.predict(model, X, y_labels)))

if __name__ == "__main__":
    main()