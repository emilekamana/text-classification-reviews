from transformers import pipeline
import streamlit as st
from utils import id2label, label2id

if 'classifier' not in st.session_state:
    classifier = pipeline("sentiment-analysis", model="emilek/reviews-text-classification")
    st.session_state['classifier'] = classifier
else:
    classifier = st.session_state['classifier']

st.title("🗨️ Sentiment and Text Classification for Product Rating Optimization 🗨️")

# Streamlit app interface
st.write("### Enter your review text below to predict the product rating:")
user_input = st.text_area("Review Text", placeholder="Type here...")


if st.button("Predict Rating"):
    if not user_input:  # Check if user_input is empty
        st.error("Please enter a review before predicting the rating.")
    else:
        result  = classifier(user_input)
        label = result[0]['label']
        rating = label2id[label] + 1

        st.write(f"Predicted Rating: {rating} {label}  { '👎' if rating<=2 else '👎'}")

