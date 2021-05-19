import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
#os.chdir('D:/dersler\Machine Learning/Machine Learning Projects')
dataset1 = pd.read_csv('SosyalMedyaReklamKampanyasi.csv')
x1 = dataset1.iloc[:,[2,3]].values
y1 = dataset1.iloc[:, 4].values


#ilk datasette bağımsız değişken olarak sadece yaş ve maası kullanacağız


#veriyi train ve test setlere ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.25, random_state=0)


#normalizasyon işlemleri
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


#SVM modelini oluşturuyoruz ve training yapıyoruz

classifier = SVC(kernel='rbf', random_state=0)

classifier.fit(X_train, y_train)

#prediction
y_pred = classifier.predict(X_test)

#hata matrisini oluşturuyoruz
cm = confusion_matrix(y_test, y_pred)
print(cm)#7 adet hata var



X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Social Media Training Set')
plt.xlabel('Yaş')
plt.ylabel('Tahmini Maaş')
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Social Media Test Set')
plt.xlabel('Yaş')
plt.ylabel('Tahmini Maaş')
plt.legend()
plt.show()