# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

# Importing the dataset
dataset = pd.read_excel('../dataset/fake_or_real_news_rnd.xlsx')#3 - ignore ""

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  # stop words are, is, the etc. which are not needed for model
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 6308): # alt+tab to align lines in loop
    review = re.sub('[^a-zA-Z]', ' ', dataset['title'][i]+' '+dataset['text'][i]) # Cleans all except characters
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)# remove words which are used less once or twice
X = cv.fit_transform(corpus).toarray()
#feature_names = cv.fit_transform(corpus).get_feature_names() # get the feature names as numpy array

'''
# Latent Semantic Analysus (LSA) Alternatively can use TF-IDF vectorizer - Feature extraction
#https://www.youtube.com/watch?v=BJ0MnawUpaU&list=PL41oaFPCrLT_lGAs7mlzq9-6N_32203HX
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))
Xv = vectorizer.fit_transform(corpus)
print(Xv[0])
'''
#gensim
#distence between topic - cross entropy distance
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
dataset.iloc[:, 2] = labelencoder_Y.fit_transform(dataset.iloc[:, 2])
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB   # alternatively MultinomialNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#serializing our model to a file called model.pkl
pkl.dump(classifier, open("news-model.pkl","wb"))
pkl.dump(cv, open("news-countvectorizer.pkl","wb"))



#loading a model from a file called model.pkl
modelx = pkl.load(open("news-model.pkl","rb"))
def format_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

new_review = format_review("""U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sunday‚Äôs unity march against terrorism. 

Kerry said he expects to arrive in Paris Thursday evening, as he heads home after a week abroad. He said he will fly to France at the conclusion of a series of meetings scheduled for Thursday in Sofia, Bulgaria. He plans to meet the next day with Foreign Minister Laurent Fabius and President Francois Hollande, then return to Washington.

The visit by Kerry, who has family and childhood ties to the country and speaks fluent French, could address some of the criticism that the United States snubbed France in its darkest hour in many years.

The French press on Monday was filled with questions about why neither President Obama nor Kerry attended Sunday‚Äôs march, as about 40 leaders of other nations did. Obama was said to have stayed away because his own security needs can be taxing on a country, and Kerry had prior commitments.

Among roughly 40 leaders who did attend was Israeli Prime Minister Benjamin Netanyahu, no stranger to intense security, who marched beside Hollande through the city streets. The highest ranking U.S. officials attending the march were Jane Hartley, the ambassador to France, and Victoria Nuland, the assistant secretary of state for European affairs. Attorney General Eric H. Holder Jr. was in Paris for meetings with law enforcement officials but did not participate in the march.

Kerry spent Sunday at a business summit hosted by India‚Äôs prime minister, Narendra Modi. The United States is eager for India to relax stringent laws that function as barriers to foreign investment and hopes Modi‚Äôs government will act to open the huge Indian market for more American businesses.

In a news conference, Kerry brushed aside criticism that the United States had not sent a more senior official to Paris as ‚Äúquibbling a little bit.‚Äù He noted that many staffers of the American Embassy in Paris attended the march, including the ambassador. He said he had wanted to be present at the march himself but could not because of his prior commitments in India.

‚ÄúBut that is why I am going there on the way home, to make it crystal clear how passionately we feel about the events that have taken place there,‚Äù he said.

‚ÄúAnd I don‚Äôt think the people of France have any doubts about America‚Äôs understanding of what happened, of our personal sense of loss and our deep commitment to the people of France in this moment of trauma.‚Äù""")

test_corpus = []
test_corpus.append(new_review)
#test_corpus.append(corpus[2])
# Creating the Bag of Words model
countvec = pkl.load(open("news-countvectorizer.pkl", 'rb'))
#from sklearn.feature_extraction.text import CountVectorizer
#loaded_vectorizer = CountVectorizer(max_features = 1500, vocabulary=vocabulary_to_load)
#loaded_vectorizer._validate_vocabulary()
X_new_test = countvec.transform(test_corpus).toarray()
prediction1 = modelx.predict(X_new_test)

