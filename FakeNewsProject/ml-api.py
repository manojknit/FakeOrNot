#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:14:50 2018

@author: Manoj
"""
#run------- $ FLASK_APP=ml-api.py FLASK_DEBUG=1 flask run
import pickle
from flask import request
from flask import Flask, url_for, render_template

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# new line remove http://removelinebreaks.net/
#{"news":"ewew."}
#code which helps initialize our server
app = Flask(__name__)

@app.route("/<name>")
def hello(name):
    return render_template('TestPage.html', name=name) #"Hello World!"

    


def format_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

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
    return pred + str(prediction) + " ~ " + outputstr
    
    
    #our model rates the wine based on the input array
    #prediction = model1.predict([feature_array]).tolist()
    
    #preparing a response object and storing the model's predictions
    #response = {}
    #response['predictions'] = prediction
    
    #sending our response object back as json
    #return Flask.jsonify(response)
    #return "h1"
    
if __name__ == "__main__":
    app.run(debug=True)
        