import os
import glob
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 載入資料集，從每個人的資料集中隨機取5張作為train data，並將資料集轉換為一維向量
def loadDataset(dir_name='./att_faces'):
    sampleNum = 5
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(40):
        img_path = dir_name+'/s'+str(i+1)+'/'
        sampled = random.sample(range(10), sampleNum)
        data = [cv.imread(d, 0) for d in glob.glob(os.path.join(img_path, '*.pgm'))]
        X_train.extend([data[i].ravel() for i in range(10) if i in sampled])
        X_test.extend([data[i].ravel() for i in range(10) if i not in sampled])
        Y_test.extend([i] * (10 - sampleNum))
        Y_train.extend([i] * sampleNum)
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

# 將資料做PCA轉換矩陣，並降維
def pca_process(X_train, Y_train, X_test, dimension):
    pca = PCA(n_components=dimension)
    pca.fit(X_train, Y_train)
    X_train_trans = pca.transform(X_train)
    X_test_trans = pca.transform(X_test)
    return X_train_trans, X_test_trans

# 將資料做FLD(LDA)轉換矩陣
def lda_process(X_train, X_test, Y_train):
    lda = LDA()
    X_train_trans = lda.fit_transform(X_train, Y_train)
    X_test_trans = lda.transform(X_test)
    return X_train_trans, X_test_trans

# 以隨機森林分類器來進行辨識，並取得預測的準確度
def rfc_model(dimension, X_train, X_test, Y_train, Y_test):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return pred

# 將混淆矩陣以pyplot顯示出來
def show_matrix_in_plot(name, dimension, con_mat):
    plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.autumn)
    plt.title('{} Dimension {} Confusion matrix'.format(name, dimension))
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name+'_'+str(dimension)+'.png')
    
def main(dimension):
    # 只做PCA的部分
    X_train, Y_train, X_test, Y_test = loadDataset()
    X_train_trans, X_test_trans = pca_process(X_train, Y_train, X_test, dimension)
    predict = rfc_model(dimension, X_train_trans, X_test_trans, Y_train, Y_test)
    print('PCA Dimension '+str(dimension)+': Accuracy: '+str((predict == np.array(Y_test)).mean() * 100))
    con_mat = confusion_matrix(predict, list(Y_test))
    show_matrix_in_plot('PCA', dimension, con_mat)
    
    # 做PCA+FLD的部分
    X_train_trans, X_test_trans = lda_process(X_train_trans, X_test_trans, Y_train)
    predict = rfc_model(dimension, X_train_trans, X_test_trans, Y_train, Y_test)
    print('LDA+PCA Dimension '+str(dimension)+': Accuracy: '+str((predict == np.array(Y_test)).mean() * 100))
    con_mat = confusion_matrix(predict, list(Y_test))
    show_matrix_in_plot('PCA+FLD(LDA)', dimension, con_mat)
    
main(10)
main(20)
main(30)
main(40)
main(50)