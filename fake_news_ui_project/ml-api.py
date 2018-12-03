#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:14:50 2018

@author: Manoj
"""
#run------- $ FLASK_APP=ml-api.py FLASK_DEBUG=1 flask run
# docker https://github.com/shekhargulati/python-flask-docker-hello-world
import pickle
from flask import request
from flask import Flask, url_for, render_template

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import string
from flask_cors import CORS
#https://stackoverflow.com/questions/29458548/can-you-add-https-functionality-to-a-python-flask-web-server

# new line remove http://removelinebreaks.net/
#{"news":"ewew."}
#code which helps initialize our server
app = Flask(__name__)
CORS(app)

cachedStopWords = set(stopwords.words('english'))

@app.route("/")
def hello():
    return render_template('TestPage.html', name='app') #"Hello World!"

def format_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in cachedStopWords]
    review = ' '.join(review)
    return review

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in cachedStopWords]


@app.route('/predict', methods=['POST'])
def predict():
    #grabbing a set of wine features from the request's body
    #feature_array = request.get_json()['feature_array']
    
    #grabs the data tagged as 'name'
    new_review = request.get_json()['news']
    outputstr = new_review;
    
    new_review = format_review(new_review)
    #return "Hello " + new_review 
    test_corpus = []
    test_corpus.append(new_review)
    # Load Bag of Words model
    cv = pickle.load(open("news-countvectorizer.pkl", 'rb'))
    X_new_test = cv.transform(test_corpus).toarray()
    
    #getting our trained model from a file we created earlier
    classifier = pickle.load(open("news-model.pkl","rb"))
    prediction = classifier.predict(X_new_test)
    pred = ""
    #return str(prediction)+" ~ "+outputstr
    if prediction == 1:
        pred= "Real"    
    else:        
        pred= "Fake"
    return pred + str(prediction) + "~" + outputstr
    
    
    #our model rates the wine based on the input array
    #prediction = model1.predict([feature_array]).tolist()
    
    #preparing a response object and storing the model's predictions
    #response = {}
    #response['predictions'] = prediction
    
    #sending our response object back as json
    #return Flask.jsonify(response)
    #return "h1"
    
@app.route('/spam', methods=['POST'])
def spam():
    #grabs the data tagged as 'name'
    new_review = request.get_json()['news']
    
    #new_review = "Trump favors imposing tariffs to protect US auto industry."
    
    new_review = format_review(new_review)
    #return "Hello " + new_review 
    test_corpus = []
    test_corpus.append(new_review)
    
    #getting our trained model from a file we created earlier
    load_model = pickle.load(open('pickle/spam_factor_model.sav', 'rb'))
    prediction = load_model.predict(test_corpus)
    prob = load_model.predict_proba(test_corpus)
    return str(prediction[0]) + "~" + str(round(prob[0,0], 2))

@app.route('/fakeness', methods=['POST'])
def fakeness():
    # https://github.com/nishitpatel01/Fake_News_Detection/blob/master/prediction.py
    #grabbing a set of wine features from the request's body
    #feature_array = request.get_json()['feature_array']
    
    #grabs the data tagged as 'name'
    #new_review = request.get_json()['news']
    new_review = request.get_json()['news']
    #outputstr = new_review;
    
    #new_review = "Trump favors imposing tariffs to protect US auto industry."

    new_review = format_review(new_review)
    #return "Hello " + new_review 
    test_corpus = []
    test_corpus.append(new_review)
    # Load Bag of Words model
    load_model = pickle.load(open('pickle/liar_liar_fact_check_factor_model.sav', 'rb'))
    prediction = load_model.predict(test_corpus)
    prob = load_model.predict_proba(test_corpus)
    #load_model.decision_function(test_corpus)
    
    classarr = load_model.classes_
    sumi = 0
    for i in range(len(classarr)):
        if "true" not in classarr[i].lower() or classarr[i].lower() == 'barely-true':
            sumi = sumi + prob[0][i]

    retout = str(prediction[0]) + "~" + str(round(sumi, 2))
    return (retout)
 
@app.route('/liarliar/<news>')
def liarliar(news):
    # https://github.com/nishitpatel01/Fake_News_Detection/blob/master/prediction.py
    #grabbing a set of wine features from the request's body
    #feature_array = request.get_json()['feature_array']
    
    #grabs the data tagged as 'name'
    #new_review = request.get_json()['news']
    new_review = news;
    #outputstr = new_review;
    
    #new_review = "Trump favors imposing tariffs to protect US auto industry."

    new_review = format_review(new_review)
    #return "Hello " + new_review 
    test_corpus = []
    test_corpus.append(new_review)
    # Load Bag of Words model
    load_model = pickle.load(open('pickle/liar_liar_fact_check_factor_model.sav', 'rb'))
    prediction = load_model.predict(test_corpus)
    prob = load_model.predict_proba(test_corpus)
    #load_model.decision_function(test_corpus)
    
    classarr = load_model.classes_
    sumi = 0
    for i in range(len(classarr)):
        if "true" not in classarr[i].lower() or classarr[i].lower() == 'barely-true':
            sumi = sumi + prob[0][i]

    retout = str(prediction[0]) + "~" + str(round(sumi, 2))
    return (retout)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
        