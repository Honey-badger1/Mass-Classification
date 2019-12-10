import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('D:\SkillShareProject\mammographic_masses.data.txt', header=None, na_values='?')
df.columns=['BI-RADS','Age','Shape','Margin','Density', 'Severity']
fill_list=[1,2,3,4,5]
df['Margin']=df['Margin'].fillna(pd.Series(np.random.choice(fill_list, size=len(df.index))))
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.corr())
df.dropna(axis=0, how='any',inplace=True)
print(df.info())
X=df[['Age','Shape','Margin','Density']].values
y=df['Severity'].values
print(X[:5])
print(y[:4])
labels=['Age','Shape','Margin','Density']

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)

from sklearn.model_selection import cross_val_score, train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y, test_size=0.25, random_state=21, stratify=y)

from sklearn import tree
from sklearn.metrics import accuracy_score
clf=tree.DecisionTreeClassifier(random_state=123)
clf = clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('DecisionTree' +':'+ str(accuracy_score(y_test,y_pred)))

scores = cross_val_score(clf, X_scaled, y, cv=10)
print('Scores'+ str(scores)) 
print(scores.mean())

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus
import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=labels)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("med.png")

 
from sklearn.ensemble import RandomForestClassifier

clf_r = RandomForestClassifier(n_estimators=10)
clf_r = clf_r.fit(X_train, y_train)
clf_r.predict(X_test)
print('RandomForest '+ str(clf_r.score(X_test, y_test)))

from sklearn import svm
clf_svc = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf_svc.predict(X_test)
print('svc_linear '+str(clf_svc.score(X_test, y_test)))
clf_svc_rbf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf_svc_rbf.predict(X_test)
print('svc_rbf '+str(clf_svc_rbf.score(X_test, y_test)))
clf_svc_sig = svm.SVC(kernel='sigmoid', C=1).fit(X_train, y_train)
clf_svc_sig.predict(X_test)
print('svc_sigmoid '+str(clf_svc_sig.score(X_test, y_test)))
clf_svc_poly = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
clf_svc_poly.predict(X_test)
print('svc_poly '+str(clf_svc_poly.score(X_test, y_test)))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.predict(X_test)
print('KNN '+ str(knn.score(X_test, y_test)))
scores_1=[]
for k in range(1,51):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    scores_1.append(knn.score(X_test, y_test))
    print('K' +':'+ str(k)+','+' score:'+ str(knn.score(X_test, y_test)))
maxs=max(scores_1)
print(maxs)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
from sklearn.preprocessing import MinMaxScaler
scalerm=MinMaxScaler()
scalerm.fit(X)
X_scaledmm=scalerm.transform(X)
X_trainm,X_testm,y_trainm,y_testm=train_test_split(X_scaledmm,y, test_size=0.25, random_state=21, stratify=y)
classifier.fit(X_trainm, y_trainm)
classifier.predict(X_testm)
sm=classifier.score(X_testm, y_testm)
print('NB '+str(sm))

