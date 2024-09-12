import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize objects
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stop_words and word not in punctuation]

    # Perform stemming
    text = [ps.stem(word) for word in text]

    transformed_text = " ".join(text)
    return transformed_text

# Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit app
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        st.write(f"Transformed text: {transformed_sms}")  # Debugging line

        vector_input = tfidf.transform([transformed_sms])
        st.write(f"Vectorized input: {vector_input.toarray()}")  # Debugging line

        result = model.predict(vector_input)[0]
        st.write(f"Prediction result: {result}")  # Debugging line

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Please enter a message to classify.")
