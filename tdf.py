import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
review = pd.read_csv('reviews.csv')
review = review.rename(columns = {'text': 'review'}, inplace = False)
review.head()
X=review.review
y=review.polarity
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.6,random_state=1)
vector = TfidfVectorizer()
# fit the vectorizer on the training data
vector.fit(X_train)
print(vector.vocabulary_)
X_transformed = vector.transform(X_train)
# for test data
X_test_transformed = vector.transform(X_test)
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)
with open('model.pkl','wb') as f:
         pickle.dump(naivebayes,f)
st.header('Sentiment Classifier')
input_text=st.text_area("Please enter the text", value="")
if st.button("Analyse"):
         input_text_transformed = vector.transform([input_text]).toarray()
         prediction = naivebayes.predict(input_text_transformed)[0]
         prediction_mapping = {0: 'NEGATIVE',1:'POSITIVE'}
         result = prediction_mapping[prediction]
         st.write(f"Predicted category: {result}")