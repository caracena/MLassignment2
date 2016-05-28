from Handwriting import HandwritingPredictor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time, logging, operator

# Create a classifier and feed into the predict function. Cross validation will be automatically used
def main():
    logging.basicConfig(filename='results.log', level=logging.INFO)

    predictor = HandwritingPredictor()
    predictor.loadFiles("semeion.data")
    predictor.displayNumbers()

    models = [svm.SVC(gamma=0.01, C=100),
              RandomForestClassifier(n_estimators=500, max_features='auto', n_jobs = -1),
              AdaBoostClassifier(n_estimators=500),
              DecisionTreeClassifier(max_features ='auto' ,max_depth=5),
              MultinomialNB(),
              GaussianNB(),
              LogisticRegression(solver='lbfgs', multi_class = 'ovr', max_iter = 50 ,n_jobs = -1),
              KNeighborsClassifier(3, n_jobs = -1),
              LinearDiscriminantAnalysis()]


    models = {}

    ## SVM configs
    svm_C = [0.1, 1, 10, 100]
    svm_gamma = ['auto', 0.01, 0.001]
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
                                                                max_depth=params[2], n_jobs = -1)

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
            results = predictor.predict(models[model])
            results_all[model] = results
            print('accuracy for model {}: {} in {} secs'.format(
                model[0],model[1], results,time.time() - start))
            logging.info('accuracy for model {}: {} in {} secs'.format(
                model[0], model[1], results, time.time() - start))
        except:
            print('error with {} and parameters {}'.format(model[0],model[1]))

    sorted_results = sorted(results_all.items(), key=operator.itemgetter(1), reverse = True)
    print('Top 10 results')
    sorted_results.take(10,sorted_results.iteritems()).foreach(lambda x : print(x))


if __name__ == "__main__":
    main()