import streamlit as st 
import joblib 
from sklearn.linear_model import LogisticRegression  # If used elsewhere

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_mdel.jb")  # Make sure the correct filename is used

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        
        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")
