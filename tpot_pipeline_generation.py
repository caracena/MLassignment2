from tpot import TPOT
import  pandas as pd
import sklearn.cross_validation as cross_validation

all_data = pd.read_csv('semeion.data', header=None,sep=' ')
x = all_data.ix[:,:255]
y = all_data.ix[:,256:265]
y_labels = [(row[row == 1].index[0] - 256) for index,row in y.iterrows()]

X_train,X_test,y_train,y_test = cross_validation.train_test_split(x, y_labels, test_size=0.33, random_state=42)

tpot = TPOT(generations=100, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_simeion_pipeline.py')