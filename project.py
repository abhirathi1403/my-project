#import all required modules
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
import pickle
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import warnings

from sklearn.utils.extmath import weighted_mode
warnings.filterwarnings('ignore')
#import dataset and save it
'''
face=fetch_lfw_people(min_faces_per_person=70)
with open('facedata1.pkl','wb') as f1:
    pickle.dump(face,f1)
'''
#import saved data from pickel file
with open('facedata1.pkl','rb') as f1:
    d=pickle.load(f1)
x=d.data
y=d.target
print(x.shape)
name=d.target_names
y=y.reshape(-1,1)
x=x/211
n_sample=x.shape[0]
features=x.shape[1]
n_classes=name.shape[0]
print("No of samples ->",n_sample)
print("no of features ->",features)
print("No of classes->",n_classes)

#split data
xtrain1,xtest1,ytrain,ytest=train_test_split(x,y,test_size=.1,random_state=50)

#feature extraction
  #using PCA
pca=PCA(n_components=.97)
xtrain=pca.fit_transform(xtrain1)
xtest=pca.transform(xtest1)
print("Features after using PCA->",xtrain.shape[1])
#hyper parameter tunning for SVC ALGO
p={'C':[.00001,.001,.005,.1,.5,1,5,7,10,100,100],'gamma':[.00001,.001,.005,.1,.5,1,5,7,10]}
clf=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid=p)
clf.fit(xtrain,ytrain)
print("best estimator ->",clf.best_estimator_)
print("score on train data->",clf.score(xtrain,ytrain))
print("Score on test data->",clf.score(xtest,ytest))
print("classification report->")
print(classification_report(ytest,clf.predict(xtest),target_names=name))

#show faces of people
b=[4,3,11,2,7,0,1]
for i in range(0,7):
  plt.subplot(2,4,i+1)
  plt.imshow(x[b[i]].reshape(62,47),cmap='gray')
  plt.title(name[i])
  plt.xticks(())
  plt.yticks(())
plt.show()