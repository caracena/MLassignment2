from Handwriting import HandwritingPredictor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
import pandas as pd
import time, logging, operator

class Main:

    def __init__(self):
        self.predictor = HandwritingPredictor()
        self.X, self.y, self.y_labels = self.predictor.loadFiles("semeion.data")

    # Create a classifier and feed into the predict function. Cross validation will be automatically used
    def getAllPredictions(self):
        logging.basicConfig(filename='results.log', level=logging.INFO)
        #predictor.displayNumbers(X,y_labels)

        models = {}

        ## SVM configs
        svm_C = [0.1, 1, 10, 100]
        svm_gamma = ['auto', 0.03, 0.003]
        svm_kernel = ['rbf', 'linear', 'poly', 'sigmoid']
        svm_parameters = [(x, y, z) for x in svm_C for y in svm_gamma for z in svm_kernel]

        for params in svm_parameters:
            models['SVM',params] = svm.SVC(gamma=params[1], C=params[0], kernel=params[2])

        ## random forest configs
        rf_nestimators = [10, 100, 300, 500]
        rf_max_features = ['auto', 'sqrt', 'log2']
        rf_max_depth =  [None, 5]
        rf_parameters = [(x, y, z) for x in rf_nestimators for y in rf_max_features for z in rf_max_depth]

        for params in rf_parameters:
            models['RandomForest', params] = RandomForestClassifier(n_estimators=params[0],
                                                                    max_features=params[1],
                                                                    max_depth=params[2], n_jobs = 4)

        ## adaboost configs
        ab_nestimators = [10, 100, 300, 500]
        ab_learning_rate = [0.1, 0.3, 1]
        ab_base_estimator = [DecisionTreeClassifier(max_depth=2, max_features ='auto'),
                             DecisionTreeClassifier(max_depth=5, max_features ='auto'),
                             DecisionTreeClassifier(max_features='auto')]
        ab_parameters = [(x, y, z) for x in ab_nestimators for y in ab_learning_rate for z in ab_base_estimator]

        for params in ab_parameters:
            models['AdaBoost', params] = AdaBoostClassifier(n_estimators = params[0], learning_rate= params[1],
                                                            base_estimator=ab_base_estimator[2])

        ## decisiontrees configs
        dt_max_depth = [None, 2, 5]
        dt_max_features = ['auto', 'sqrt', 'log2']
        dt_parameters = [(x, y) for x in dt_max_depth for y in dt_max_features]

        for params in dt_parameters:
            models['DecisionTrees', params] = DecisionTreeClassifier(max_depth=params[0], max_features=params[1])

        ## MutinomialNB configs
        mnb_aplpha = [0.1, 0.3, 1]

        for params in mnb_aplpha:
            models['MultinomialNB', params] = MultinomialNB(alpha=params)

        ## GaussianNB configs
        models['GaussianNB'] = GaussianNB()

        ## LogisticRegression configs
        lr_C = [0.1, 1, 10, 100]
        lr_multi_class = ['ovr', 'multinomial']
        lr_parameters = [(x, y) for x in lr_C for y in lr_multi_class]

        for params in lr_parameters:
            models['LogisticRegression', params] = LogisticRegression(C=params[0], multi_class=params[1], n_jobs=-1)

        ## KNeighborsClassifier configs
        knn_n_neighbors = [3, 5, 7]
        knn_p = [1, 2, 3]
        knn_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        knn_paramters = [(x, y, z) for x in knn_n_neighbors for y in knn_p for z in knn_algorithm]

        for params in knn_paramters:
            models['KNeighbors', params] = KNeighborsClassifier(n_neighbors=params[0], p=params[1], algorithm=params[2], n_jobs=-1)


        ## LinearDiscriminantAnalysis configs
        lda_solver = ['svd', 'lsqr', 'eigen']
        lda_n_components = [3, 5, 8]
        lda_parameters = [(x, y) for x in lda_solver for y in lda_n_components]

        for params in lda_parameters:
            models['LinearDiscriminantAnalysis', params] = LinearDiscriminantAnalysis(solver=params[0], n_components=params[1])


        results_all = {}

        for model in models.keys():
            try:
                start = time.time()
                results = self.predictor.predict(models[model],self.X,self.y_labels)
                results_all[model] = results
                #print('accuracy for model {}: {} in {} secs'.format(model[0],model[1], results,time.time() - start))
                logging.info('accuracy for model {}: {} in {} secs'.format(
                    model[0], model[1], results, time.time() - start))
            except:
                print('error with {} and parameters {}'.format(model[0],model[1]))

        sorted_results = sorted(results_all.items(), key=operator.itemgetter(1), reverse = True)
        [logging.info(x) for x in sorted_results]


    def PCA(self,n_components=0.9):
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(self.X)
        X_reduced = pd.DataFrame(pca.fit_transform(self.X))
        X_reconstructed = pca.inverse_transform(X_reduced)
        return X_reduced,X_reconstructed

    def runPredictionByModel(self,model,X,y):
        return self.predictor.predict(model,X,y)

if __name__ == "__main__":
    m = Main()
    print("### Running all models")
    m.getAllPredictions()

    print("### Running 2 best models")

    print("### Running SVM model with params (gamma 0.03, C=100)")
    start = time.time()
    clf = svm.SVC(gamma=0.03, C=100)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf,m.X,m.y_labels),time.time() - start))

    print("### Running SVM model with params (gamma 0.03, C=100) and PCA with 90% of variance")
    start = time.time()
    X_reduced_90,X_reconstructed_90 = m.PCA(0.9)
    clf = svm.SVC(gamma=0.03, C=100)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf,X_reduced_90,m.y_labels),time.time() - start))

    print("### Running SVM model with params (gamma 0.03, C=100) and PCA with 50% of variance")
    start = time.time()
    X_reduced_50,X_reconstructed_50 = m.PCA(0.5)
    clf = svm.SVC(gamma=0.03, C=100)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf,X_reduced_50,m.y_labels),time.time() - start))

    print("### Running SVM model with params (gamma 0.03, C=100) and PCA with 10% of variance")
    start = time.time()
    X_reduced_10,X_reconstructed_10 = m.PCA(0.1)
    clf = svm.SVC(gamma=0.03, C=100)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf,X_reduced_10,m.y_labels),time.time() - start))

    print("### Running RandomForest model with params (n_estimators=500, max_features=log2)")
    start = time.time()
    clf = RandomForestClassifier(500,max_features="log2",max_depth=None)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf, m.X, m.y_labels), time.time() - start))

    print("### Running RandomForest model with params (n_estimators=500, max_features=log2) and PCA with 90% of variance")
    start = time.time()
    clf = RandomForestClassifier(500,max_features="log2",max_depth=None)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf, X_reduced_90, m.y_labels), time.time() - start))

    print("### Running RandomForest model with params (n_estimators=500, max_features=log2) and PCA with 50% of variance")
    start = time.time()
    clf = RandomForestClassifier(500,max_features="log2",max_depth=None)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf, X_reduced_50, m.y_labels), time.time() - start))

    print("### Running RandomForest model with params (n_estimators=500, max_features=log2) and PCA with 10% of variance")
    start = time.time()
    clf = RandomForestClassifier(500,max_features="log2",max_depth=None)
    print("Score: {}, Total Time (s): {}".format(m.runPredictionByModel(clf, X_reduced_10, m.y_labels), time.time() - start))


