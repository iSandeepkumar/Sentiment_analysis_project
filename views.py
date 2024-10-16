from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import time
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def preprocess():
    textdata.clear()
    labels.clear()
    dataset = pd.read_csv('Dataset/reviews.csv')
    for i in range(len(dataset)):
        msg = dataset._get_value(i, 'Review')
        label = dataset._get_value(i, 'Label')
        msg = str(msg)
        msg = msg.strip().lower()
        lab = int(label)
        if lab >= 3:
            labels.append(1)
        else:
            labels.append(0)
        clean = cleanPost(msg)
        textdata.append(clean)
    TFIDFfeatureEng()    

def TFIDFfeatureEng():
    global Y
    global X
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=100)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    print(str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:100]
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    
def TestSentiment(request):
    if request.method == 'GET':
       return render(request, 'TestSentiment.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Admin(request):
    if request.method == 'GET':
       return render(request, 'Admin.html', {})

def AdminLogin(request):
    if request.method == 'POST':
      username = request.POST.get('t1', False)
      password = request.POST.get('t2', False)
      if username == 'admin' and password == 'admin':
       context= {'data':'welcome '+username}
       return render(request, 'AdminScreen.html', context)
      else:
       context= {'data':'login failed'}
       return render(request, 'Admin.html', context)

def Upload(request):
    if request.method == 'GET':
       return render(request, 'Upload.html', {})


def UploadDataset(request):
    if request.method == 'POST' and request.FILES['t1']:
        output = ''
        myfile = request.FILES['t1']
        preprocess()
        output+="Total Reviews found in dataset : "+str(len(X))+"<br>"
        output+="Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"<br>"
        output+="Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"<br>"
        context= {'data':output}
        return render(request, 'Upload.html', context)
    
def kmeanssvm(request):
    if request.method == 'GET':
        start = time.time()
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)
        predict = kmeans.predict(tfidf_X_test)
        kmeans_acc = accuracy_score(tfidf_y_test,predict)*100
        end = time.time()
        kmeans_time = end - start

        start = time.time()
        cls = svm.SVC() 
        cls.fit(X, Y)
        predict = cls.predict(tfidf_X_test)
        svm_acc = (accuracy_score(tfidf_y_test,predict)) * 100
        end = time.time()
        svm_time = end - start

        output = 'KMEANS Accuracy : '+str(kmeans_acc)+" KMEANS Execution Time : "+str(kmeans_time)+"<br/>"
        output+='SVM Accuracy : '+str(svm_acc)+" SVM Execution Time : "+str(svm_time)+"<br/>"

        height = [svm_acc, kmeans_acc, svm_time, kmeans_time]
        bars = ('SVM Accuracy','KMeans Accuracy','SVM Time','KMEANS Time')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.show()
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)


def knnkmeans(request):
    if request.method == 'GET':
        start = time.time()
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)
        predict = kmeans.predict(tfidf_X_test)
        kmeans_acc = accuracy_score(tfidf_y_test,predict)*100
        end = time.time()
        kmeans_time = end - start

        start = time.time()
        cls = KNeighborsClassifier(n_neighbors = 2) 
        cls.fit(X, Y)
        predict = cls.predict(tfidf_X_test)
        knn_acc = (accuracy_score(tfidf_y_test,predict)) * 100
        end = time.time()
        knn_time = end - start

        output = 'KMEANS Accuracy : '+str(kmeans_acc)+" KMEANS Execution Time : "+str(kmeans_time)+"<br/>"
        output+='KNN Accuracy : '+str(knn_acc)+" KNN Execution Time : "+str(knn_time)+"<br/>"

        height = [knn_acc, kmeans_acc, knn_time, kmeans_time]
        bars = ('KNN Accuracy','KMeans Accuracy','KNN Time','KMEANS Time')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.show()
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)


def nbcnn(request):
    if request.method == 'GET':
        start = time.time()
        cls = MultinomialNB() 
        cls.fit(X, Y)
        predict = cls.predict(tfidf_X_test)
        nb_acc = (accuracy_score(tfidf_y_test,predict)) * 100
        end = time.time()
        nb_time = end - start
        Y1 = to_categorical(Y)
        start = time.time()
        cnn_model = Sequential()
        cnn_model.add(Dense(512, input_shape=(tfidf_X_train.shape[1],)))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(512))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(2))
        cnn_model.add(Activation('softmax'))
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(cnn_model.summary())
        acc_history = cnn_model.fit(X, Y1, epochs=10)
        print(cnn_model.summary())
        acc_history = acc_history.history
        acc_history = acc_history['acc']
        cnn_acc = acc_history[9] * 100
        end = time.time()
        cnn_time = end - start

        output = 'Naive Bayes Accuracy : '+str(nb_acc)+" Naive Bayes Execution Time : "+str(nb_time)+"<br/>"
        output+='CNN Accuracy : '+str(cnn_acc)+" CNN Execution Time : "+str(cnn_time)+"<br/>"

        height = [cnn_acc, nb_acc, cnn_time, nb_time]
        bars = ('CNN Accuracy','Naive Bayes Accuracy','CNN Time','Naive Bayes Time')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.show()
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)    


def DetectSentiment(request):
    if request.method == 'POST':
        sentence = request.POST.get('t1', False)
        msg = sentence
        review = sentence.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        sentiment_dict = sid.polarity_scores(review)
        negative = sentiment_dict['neg']
        positive = sentiment_dict['pos']
        neutral = sentiment_dict['neu']
        compound = sentiment_dict['compound']
        result = ''
        if compound >= 0.05 : 
            result = 'Positive' 
  
        elif compound <= - 0.05 : 
            result = 'Negative' 
  
        else : 
            result = 'Neutral'
    
        output = msg+'<br/>CLASSIFIED AS '+result+"<br/>"
        plt.pie([positive,negative,neutral],labels=["Positive","Negative","Neutral"],autopct='%1.1f%%')
        plt.title('Sentiment Graph')
        plt.axis('equal')
        plt.show()
        context= {'data':output}
        return render(request, 'TestSentiment.html', context)    










        
