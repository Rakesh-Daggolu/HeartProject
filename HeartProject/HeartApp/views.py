from django.shortcuts import render
from django.shortcuts import redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from HeartApp.forms import *
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_regression
#import base64
#from io import BytesIO
from HeartApp.utils import *
from sklearn.metrics import accuracy_score
from sklearn import svm,preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from HeartApp.outer import *
# Create your views here.

class OuterObj:
   '''
     Documentation String
     This class used to access various functions of outer module
   '''
   outer=""
   @classmethod
   def set_outer(self):
      self.outer=Outer()
      return self.outer

   @classmethod
   def get_outer(self):
       return self.outer

#Home Page
def Home(request):
    return render(request,'HeartApp/base.html')

#Abstract Page
def Abstract(request):
    if request.method=='POST':
        return redirect('/upload')
    return render(request,'HeartApp/abstract.html')


#Uploading The DataSet
def Upload(request):
    global uploaded_file_url
    if request.method=='POST':
        doc=request.FILES.get('document')
        fs = FileSystemStorage()
        filename = fs.save(doc.name,doc)
        uploaded_file_url = fs.url(filename)
        return redirect('/display/?filename='+filename)
    return render(request,'HeartApp/upload.html')

#Displaying The DataSet
def Display(request):
    #global filename
    filename=request.GET.get('filename')
    f=open('./media/'+filename,'r')
    w=csv.reader(f)
    data=list(w)
    f.close()
    df = pd.read_csv('./media/'+filename)
    df.drop_duplicates(keep=False)

    if request.method=="POST":
        OuterObj().set_outer()
        normalized_data,matrix_results=getresults(df)
        OuterObj().get_outer().set_normalizeddata(normalized_data)
        import os
        if os.path.exists('./media/'+filename):
           os.remove('./media/'+filename)
        return render(request,'HeartApp/plot.html',{'matrix_results':matrix_results})

    #print(len(df))
    #print(type(df))
    sns.countplot(x='target',data=df)
    fig=plt.savefig('./static/img/statistic.png')
    plt.close()

    return render(request,'HeartApp/display.html',{'data':data})


#Data Preprocessing
def getresults(df):
    d1={0:'age',1:'sex',2:'cp',3:'trestbps',4:'chol',5:'fbs',6:'restecg',7:'thalach',8:'exang',9:'oldpeak',10:'slope',11:'ca',12:'thal',13:
    'target'}
    X = df.iloc[:,0:13].values
    y = df.iloc[:,13].values
    #print(len(X))
    #s1=df.isna().sum()
    #print(s1)
    #s2=s1.tolist()
    #print(s2)
    dfNorm,normalized_data=normalize(X,df)
    draw_hist(dfNorm)
    dfNorm['target'] = df['target']
    corr_matrix_results=save_correlation_matrix(dfNorm,d1)
    OuterObj().get_outer().set_dfNorm(dfNorm)
    OuterObj().get_outer().set_X(X)
    #print(y)
    OuterObj().get_outer().set_Y(y)
    OuterObj().get_outer().set_d1(d1)
    return normalized_data,corr_matrix_results

#Normalize The DataSet df
def normalize(X,df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    #print(type(scaled),df.index)                 #  numpy.ndarray,(start=0,stop=1025,step=1)
    dfNorm = pd.DataFrame(scaled,index=df.index, columns=df.columns[0:13])
    normalized_data=dfNorm.head(10).values.tolist()
    #print(dfNorm.tail(10).values.tolist())
    return dfNorm,normalized_data

#Save The CorrelationMatrix In the static folder
def save_correlation_matrix(dfNorm,d1):
    corr_mat=dfNorm.corr()
    corr_matrix_results=get_cols(corr_mat.values.tolist(),d1)
    plt.figure(figsize=(15,5))                    # (float,float)
    plt.title("Correlation Matrix")
    #print(corr_mat.columns)
    svm=sns.heatmap(corr_mat,xticklabels=corr_mat.columns,yticklabels=corr_mat.columns,linewidths=1,annot=True)
    plt.savefig('./static/correlationmatrix.png')
    plt.close()
    return corr_matrix_results

#What We Infered From Correlation Matrix
def get_cols(df,d):
    n=len(df)
    l=[]
    for i in range(n):
        pos=-2
        neg=2
        indexpos=0
        indexneg=0
        for j in range(n):
            if i!=j:
                if df[i][j]>0:
                   if pos<df[i][j]:
                       pos=df[i][j]
                       indexpos=j
                else:
                     if neg>df[i][j]:
                        neg=df[i][j]
                        indexneg=j

        l.append((d[i],d[indexpos],pos,d[indexneg],neg))
    return l

#Save The Histograms In the Static Folder
def draw_hist(dfNorm):
    dfNorm.hist(figsize=(10,10))
    plt.savefig('./static/histogram.png')
    plt.close()

#Display The Normalized Data Of Top 10 Records In The Html Page
def Preprocess(request):
    normalized_data=OuterObj().get_outer().get_normalizeddata()
    return render(request,'HeartApp/preprocess.html',{'normalized_data':normalized_data})

#Split and Data Set Without Feature Selection
def WithoutFeatureSelection(request):
    dfNorm,X,y,d1=get_dfNorm_X_y_d1()
    OuterObj().get_outer().set_P(X)
    y=dfNorm['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=00)
    set_Xtrain_Xtest_Ytrain_Ytest(X_train, X_test, y_train, y_test)
    return render(request,'HeartApp/fcmim.html',{'b':False})

#Relief Feature Selection Algorithm
def Relif(request):
    dfNorm,X,y,d1=get_dfNorm_X_y_d1()
    X1=relif(dfNorm,X,y,d1)
    OuterObj.get_outer().set_P(X1)
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20, random_state=0)
    set_Xtrain_Xtest_Ytrain_Ytest(X_train, X_test, y_train, y_test)
    return render(request,"HeartApp/relif.html")

#FCMIM Feature Selection Algorithm
def Fcmim(request):
    dfNorm,X,y,d1=get_dfNorm_X_y_d1()
    X1=perform_fcmim(dfNorm,X,y,d1)
    OuterObj().get_outer().set_P(X1)
    y=dfNorm['target']
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20,random_state=00)
    set_Xtrain_Xtest_Ytrain_Ytest(X_train, X_test, y_train, y_test)
    return render(request,'HeartApp/fcmim.html',{'b':True})

#MRMR Feature Selection Algorithm
def MRMR(request):
    from mrmr import mrmr_classif
    from sklearn.datasets import make_classification
    dfNorm,X,y,d1=get_dfNorm_X_y_d1()
    selected_features = mrmr_classif(pd.DataFrame(X), pd.Series(y), K = 11)
    col=[d1[i] for i in selected_features]
    X1=dfNorm[col]
    OuterObj().get_outer().set_P(X1)
    y=dfNorm['target']
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20,random_state=00)
    set_Xtrain_Xtest_Ytrain_Ytest(X_train, X_test, y_train, y_test)
    return render(request,'HeartApp/mrmr.html',{'col':col})

#Return The Normalization,X,y,d1
def get_dfNorm_X_y_d1():
    obj=OuterObj().get_outer()
    return obj.get_dfNorm(),obj.get_X(),obj.get_Y(),obj.get_d1()

#Set The X_train,X_test,y_train,y_test
def set_Xtrain_Xtest_Ytrain_Ytest(X_train, X_test, y_train, y_test):
    obj=OuterObj().get_outer()
    obj.set_Xtrain(X_train)
    obj.set_Xtest(X_test)
    obj.set_Ytrain(y_train)
    obj.set_Ytest(y_test)

#claculate distance between entities using manhattan distance
def distance(p,q):
    d=0
    for i in range(len(p)):
        d+=abs(p[i]-q[i])
    return d

#Relif Algorithm
def relif(df,X,y,d):
    w={}
    for u in d.keys():
       w[u]=0
    l=pd.DataFrame(X).values.tolist()
    att=[]
    for i in range(13):
      m=inf
      m1=-inf
      for j in range(len(l)):
         m=min(m,l[j][i])
         m1=max(m1,l[j][i])
      att.append(abs(m1-m))
      print((l[0][i]-m)/att[i],end=' ')   #print normalized_data for 1st row

    #n=len(l)-1
    m=11
    j=len(l)-1
    for i in range(m):
       R=l[j]
       target_type=y[j]
       H=inf
       M=inf
       H_row=-1
       M_row=-1
       for k in range(len(l)):
           if k!=j:
               if target_type==y[k]:
                   if H>distance(R,l[k]):
                       H=distance(R,l[k])
                       H_row=l[k]
               else:
                   if M>distance(R,l[k]):
                      M=distance(R,l[k])
                      M_row=l[k]
       for k in range(13):
             v1=0
             v2=0
             v1=(abs(R[k]-H_row[k])/att[k])/m
             v2=(abs(R[k]-M_row[k])/att[k])/m
             w[k]=w[k]-v1+v2
             print(w[k])
       j-=1
    l=[]
    for u,v in w.items():
        l.append([u,v])
    print()
    print(l)
    l.pop()
    l.sort(key=lambda x:x[1])
    for i in range(13-m):
            l.pop()
    l.sort(key=lambda x:x[0])
    col=[d[x[0]] for x in l]
    b=[x[1] for x in l]
    X = df[col]
    plt.figure(figsize=(10,5))
    plt.bar(col,b, color ='maroon',width = 0.4)
    plt.title("Relif features selected")
    plt.savefig('./static/relif.png')
    plt.close()
    return X

#Intialize The No Of Parameters and Display Their Feature Scores In The Form Of BarGraph
def perform_fcmim(dfNorm,X,y,d1):
    d={'n_selected_features':11}
    a,b=fast_cmim(X,y,d)
    for i in range(len(a)):
        for j in range(len(a)-i-1):
            if a[j]>a[j+1]:
                a[j+1],a[j]=a[j],a[j+1]
                b[j+1],b[j]=b[j],b[j+1]
    col=[d1[i] for i in a]
    OuterObj().get_outer().set_col(col)
    print(col)
    X = dfNorm[col]
    plt.figure(figsize=(10,5))
    plt.bar(col,b, color ='maroon',width = 0.4)
    plt.title("Feature Selected by FCMIM")
    plt.savefig('./static/feature.png')
    plt.close()
    return X


#Test For The Customised Data
def Test(request):
    form=HeartForm()
    if request.method=='POST':
        form=HeartForm(request.POST)
        if form.is_valid():
            data=form.cleaned_data
            col=OuterObj.get_outer().get_col()
            d1={'age':0,'sex':1,'cp':2,'trestbps':3,'chol':4,'fbs':5,'restecg':6,'thalach':7,'exang':8,'oldpeak':9,'slope':10,'ca':11,'thal':12,13:
            'target'}
            value=[]
            for u,v in data.items():
                if u in col:
                   value.append([d1[u],v])

            value.sort(key=lambda x:x[0])
            l=[]
            for x in value:
                l.append(x[1])
            #print(l)
            l=np.asarray(l)
            print(l)
            l_reshaped=l.reshape(1,-1)
            #print(l_reshaped)
            SVC_classifier()
            return render(request,'HeartApp/test.html',{'form':form,'lable':SVC_classifier(l_reshaped,True),'b':True})
    return render(request,'HeartApp/test.html',{'form':form,'b':False})


#Compare The Results Of Various Classifiers
def Compare(request):
    l=[]
    k=["KNN","SVC","LR","NB","DT","ANN"]
    f=[KNN_classifier,SVC_classifier,Logistic_Regression,NavieBayes,DesicionTree,ANN_classifier]
    for i in range(6):
        l.append((k[i],)+f[i]())
    return render(request,'HeartApp/compare.html',{'l':l})


#Display accuracy,sensitivity,specificity,precision in html page
def Test1(request):
    accuracy=OuterObj().get_outer().get_accuracy()
    sensitivity=OuterObj().get_outer().get_sensitivity()
    specificity=OuterObj().get_outer().get_specificity()
    precision=OuterObj().get_outer().get_precision()
    return render(request,'HeartApp/test1.html',{'accuracy':round(accuracy,3),'sensitivity':round(sensitivity,3),'specificity':round(specificity,3),'precision':round(precision,3)})


#Support Vector classification
def SVC_classifier(values=None,b=False):
    from sklearn import svm
    X_train,X_test,y_train,y_test=get_Xtrain_Xtest_Ytrain_Ytest()
    classifier=svm.SVC(kernel='linear',gamma=0.0009,C=10)
    classifier.fit(X_train,y_train)
    if b:
         return classifier.predict(values)
    else:
       y_predict=classifier.predict(X_test)
    return get_confusion_matrix(y_test,y_predict)

#Logistic Regression Classification
def Logistic_Regression():
    from sklearn.linear_model import LogisticRegression
    X_train,X_test,y_train,y_test=get_Xtrain_Xtest_Ytrain_Ytest()
    classifier= LogisticRegression(C=10,random_state=0)
    classifier.fit(X_train , y_train)
    y_predict = classifier.predict(X_test)
    return get_confusion_matrix(y_test,y_predict)

#Navie Bayes classification
def NavieBayes():
    from sklearn.naive_bayes import GaussianNB
    X_train,X_test,y_train,y_test=get_Xtrain_Xtest_Ytrain_Ytest()
    classifier= GaussianNB()
    classifier.fit(X_train , y_train)
    y_predict = classifier.predict(X_test)
    return get_confusion_matrix(y_test,y_predict)

#ANN classification
def ANN_classifier():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    import tensorflow as tf
    from tensorflow import keras
    X_train,X_test,y_train,y_test=get_Xtrain_Xtest_Ytrain_Ytest()
    dfNorm=OuterObj().get_outer().get_dfNorm()

    x_training_data = scaler.fit_transform(X_train)
    x_test_data = scaler.fit_transform(X_test)
    shape=X_train.shape[1]                           #(1025,11)
    ann= tf.keras.models.Sequential()                #building the neural network
                                                     #dense means each neuron connected to every other neuron
    ann.add(tf.keras.layers.Dense(units = shape, input_shape=(shape,))) #Input layer
    ann.add(tf.keras.layers.Dense(units = 40, activation = 'relu')) #First hidden layer
    ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) #Output layer

    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier=ann.fit(x_training_data, y_train, epochs = 2) #train the ann
    y_predict = ann.predict(x_test_data)
    y_pred=[]
    for element in y_predict:
        if element > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    from sklearn.metrics import accuracy_score
    #print(y_test,y_pred,sep='\n')
    #print(accuracy_score(y_test,y_pred))
    #Generate an accuracy score
    cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
    plt.figure(figsize=(16,9))
    plt.title("Confusion Matrix")
    sns.heatmap(cm,annot=True)
     #print(int(cm[0][0]))
    TN = int(cm[0][0])
    FP = int(cm[0][1])
    FN = int(cm[1][0])
    TP = int(cm[1][1])

    plt.savefig('./static/confusion_matrix.png')
    plt.close()
    return round(((TP+TN)/(TP+TN+FP+FN))*100,2),round((TP/(TP + FN))*100,2),round(TN/(TN + FP)*100,2),round(TP/(TP + FP)*100,2)


#DesicionTree classification
def DesicionTree():
    from sklearn.tree import DecisionTreeClassifier
    X1=OuterObj().get_outer().get_P()
    y=OuterObj().get_outer().get_Y()
    #print(type(X1),type(y))
    classifier= DecisionTreeClassifier()
    X_train1=X1[900:1026]
    X_test1=X1[0:800]
    y_train1=y[900:1026]
    y_test1=y[0:800]
    classifier.fit(X_train1 , y_train1)
    y_predict = classifier.predict(X_test1)
    return get_confusion_matrix(y_test1,y_predict)

#KNN classification
def KNN_classifier():
    X_train,X_test,y_train,y_test=get_Xtrain_Xtest_Ytrain_Ytest()
    X1=OuterObj().get_outer().get_P()
    y=OuterObj().get_outer().get_Y()
    classifier = KNeighborsClassifier(n_neighbors=7)
    X_train1=X1[205:1026]
    X_test1=X1[0:205]
    y_train1=y[205:1026]
    y_test1=y[0:205]
    classifier.fit(X_train1 , y_train1)
    y_predict = classifier.predict(X_test1)
    return get_confusion_matrix(y_test1,y_predict)

#set accuracy,sensitivity,specificity,precision
def set_calculated_parameters(accuracy,sensitivity,specificity,precision):
    OuterObj().get_outer().set_accuracy(accuracy)
    OuterObj().get_outer().set_sensitivity(sensitivity)
    OuterObj().get_outer().set_specificity(specificity)
    OuterObj().get_outer().set_precision(precision)

#get X_train,X_test,y_train,y_test
def get_Xtrain_Xtest_Ytrain_Ytest():
    obj=OuterObj().get_outer()
    return obj.get_Xtrain(),obj.get_Xtest(),obj.get_Ytrain(),obj.get_Ytest()


#Display Confusion Matrix,Accuracy,Sensitivity,Specificity,Precision
def KNN(request):
    accuracy,sensitivity,specificity,precision=KNN_classifier()
    set_calculated_parameters(accuracy,sensitivity,specificity,precision)
    return Test1(request)

#Display Confusion Matrix,Accuracy,Sensitivity,Specificity,Precision For SVC
def SVC(request):
    accuracy,sensitivity,specificity,precision=SVC_classifier()
    set_calculated_parameters(accuracy,sensitivity,specificity,precision)
    return Test1(request)

#Display Confusion Matrix,Accuracy,Sensitivity,Specificity,Precision for LR
def LR(request):
    accuracy,sensitivity,specificity,precision=Logistic_Regression()
    set_calculated_parameters(accuracy,sensitivity,specificity,precision)
    return Test1(request)

#Display Confusion Matrix,Accuracy,Sensitivity,Specificity,Precision for NB
def NB(request):
    accuracy,sensitivity,specificity,precision=NavieBayes()
    set_calculated_parameters(accuracy,sensitivity,specificity,precision)
    return Test1(request)

#Display Confusion Matrix,Accuracy,Sensitivity,Specificity,Precision for DT
def DT(request):
    accuracy,sensitivity,specificity,precision=DesicionTree()
    set_calculated_parameters(accuracy,sensitivity,specificity,precision)
    return Test1(request)

#Display Confusion Matrix,Accuracy,Sensitivity,Specificity,Precision for ANN
def ANN(request):
    accuracy,sensitivity,specificity,precision=ANN_classifier()
    set_calculated_parameters(accuracy,sensitivity,specificity,precision)
    return Test1(request)

#Save the Confusion Matrix in static folder and Calculate Accuracy,Sensitivity,Specificity,Precision
def get_confusion_matrix(y_test,y_predict):
    cm=confusion_matrix(y_test,y_predict)
    plt.figure(figsize=(16,9))
    plt.title("Confusion Matrix")
    sns.heatmap(cm,annot=True)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    plt.savefig('./static/confusion_matrix.png')
    plt.close()
    return round(((TP+TN)/(TP+TN+FP+FN))*100,2),round((TP/(TP + FN))*100,2),round(TN/(TN + FP)*100,2),round(TP/(TP + FP)*100,2)


def fast_cmim(X, Y, kwargs):
    no_of_samples, no_of_features = X.shape
    is_features_specified = False
    #print(no_of_samples,no_of_features)  1025 , 13
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_features_specified = True
        F = np.nan * np.zeros(n_selected_features) #F=[nan]*11
    else:
        F = np.nan * np.zeros(no_of_features)      #F=[nan]*13

    #print(is_features_specified,F)     True,  [nan nan nan nan nan nan nan nan nan nan nan]

    T1 = np.zeros(no_of_features)       #T1=[0]*13
    M = np.zeros(no_of_features) - 1    #m=[-1]*13

    #print(T1,m)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]

    for i in range(no_of_features): #execute loop for 13 times
        #print(X[:,i])
        f = X[:, i]               #for 1st row to 13th row
        T1[i] = midd(f, Y)        #getting the mutual information of all features

    #print(T1,f)
    #[0.14284927 0.05798415 0.20848408 0.13669189 0.57275283 0.00122222,0.02556413 0.33203352 0.14508525 0.26247835 0.11294555 0.1925456 0.20777926]
    # [3. 3. 3. ... 2. 2. 3.]

    for k in range(no_of_features):
        if k == 0:        #selecting the feature with highest mutual information
            idx = np.argmax(T1)
            F[0] = idx    #F=[4 nan nan nan nan nan nan nan nan nan nan]
            f_select = X[:, idx]  #f_select=212...188 all values of column4
            #print(idx)    idx=4
            #print(f_select)

        if is_features_specified:
            #print(np.sum(~np.isnan(F)),n_selected_features)
            #np.sum(np.sum(~np.isnan(F)))  count all featuers in F not equal to nan F[i]!=nan
            if np.sum(~np.isnan(F)) == n_selected_features: #check for required no of features selected or not
                break

        sstar = -1000000
        for i in range(no_of_features):   #execute for 13 times
            if i not in F:            #0!=F,1!=F
                while (T1[i] > sstar) and (M[i]<k-1) : #t[i]>score and m[i]<k-1
                    M[i] = M[i] + 1                                               #m[1]=m[1]+1-> m[1]=0
                    T1[i] = min(T1[i], cmidd(X[:,i], # feature i                  #T1[1]=gets_updates
                                             Y,  # target
                                             X[:, int(F[int(M[i])])] # conditionned on selected features
                                            )
                               )
                if T1[i] > sstar:
                    sstar = T1[i]   #sstar=score
                    F[k+1] = i      #F[1]=0.14824

    #F = np.array(F[F>-100])
    F = F.astype(int)
    T1 = T1[F]
    return (F, T1)
