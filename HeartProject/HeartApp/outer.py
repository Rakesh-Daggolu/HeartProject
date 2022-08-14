class Outer:
    '''
    Documentation String

    This class contains various getter and setter methods which are accessed by various functions of views.py
    '''
    def __init__(self):
        self.dfNorm=""
        self.X=""
        self.y=""
        self.d1=""
        self.P=""
        self.X_train=""
        self.y_train=""
        self.X_test=""
        self.y_test=""
        self.col=""
        self.accuracy=""
        self.sensitivity=""
        self.specificity=""
        self.precision=""
        self.normalized_data=""
        self.col=""
    def set_dfNorm(self,dfNorm):
        self.dfNorm=dfNorm
    def set_X(self,X):
        self.X=X
    def set_Y(self,Y):
        self.y=Y

    def set_d1(self,d1):
        self.d1=d1
    def set_P(self,P):
        self.P=P
    def set_Xtrain(self,Xtrain):
        self.X_train=Xtrain
    def set_Ytrain(self,Ytrain):
        self.y_train=Ytrain
    def set_Xtest(self,Xtest):
        self.X_test=Xtest
    def set_Ytest(self,ytest):
        self.y_test=ytest
    def set_col(self,col):
        self.col=col
    def set_accuracy(self,accuracy):
        self.accuracy=accuracy
    def set_sensitivity(self,sensitivity):
        self.sensitivity=sensitivity
    def set_specificity(self,specificity):
        self.specificity=specificity
    def set_precision(self,precision):
        self.precision=precision
    def set_normalizeddata(self,normalized_data):
        self.normalized_data=normalized_data
    def set_col(self,col):
        self.col=col

    def get_dfNorm(self):
        return self.dfNorm
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.y
    def get_d1(self):
        return self.d1
    def get_P(self):
        return self.P
    def get_Xtrain(self):
        return self.X_train
    def get_Ytrain(self):
        return self.y_train
    def get_Xtest(self):
        return self.X_test
    def get_Ytest(self):
        return self.y_test
    def get_col(self,col):
        return self.col
    def get_accuracy(self):
        return self.accuracy
    def get_sensitivity(self):
        return self.sensitivity
    def get_specificity(self):
        return self.specificity
    def get_precision(self):
        return self.precision
    def get_normalizeddata(self):
        return self.normalized_data
    def get_col(self):
        return self.col
