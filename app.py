import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Mail Classifier")

input_message = st.text_area('Enter the message')


def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # convert string to list
    y = []

    # Special Characters
    for i in text:  # removing special characters
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Stop Words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button('Predict'):
    # 1. preprocess
    transform_message = transform_text(input_message)

    # 2. vectorize
    vectorized_input = tfidf.transform([transform_message])

    # 3. predict
    result = model.predict(vectorized_input)[0]

    # 4. display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")