import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained LSTM model and tokenizer
def load_tweet_model():
    model = load_model('final_stress_model.h5')  
    with open('trained_preprocessor.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

tweet_model, tokenizer = load_tweet_model()
max_seq_len = 50  # Ensure this matches what was used during training

def analyze_text(text):
    # Tokenize and pad the text
    text_seq = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(text_seq, maxlen=max_seq_len, padding="post", truncating="post")

    # Predict the sentiment and stress
    pred = tweet_model.predict(padded_text)
    pred_value = float(pred[0][0])

    # Determine sentiment
    if pred_value <= 0.349:
        sentiment_label = "positive"
    elif 0.35 <= pred_value <= 0.599:
        sentiment_label = "neutral"
    else:
        sentiment_label = "negative"

    # Determine stress
    stress_label = 'STRESS' if pred_value > 0.5 else 'NO-STRESS'

    return pred_value, sentiment_label, stress_label

def show_tweet_page():
    st.title("Input and Analyze Text")

    st.image('stress.png', caption='Text Analysis', use_column_width=True)


    # User input
    new_text = st.text_area('Enter a text to analyze:')

    # Button to trigger analysis
    if st.button("Analyze Text"):
        if new_text:
            pred_value, sentiment_label, stress_label = analyze_text(new_text)

            st.write(f"**Text to Analyze:** {new_text}")
            st.write(f"The Prediction Value: {pred_value:.2f}")
            st.write(f"The Predicted Sentiment: {sentiment_label}")

            # Display stress level with color and capital letters
            if stress_label == 'STRESS':
                st.markdown(
                    f"<span style='color:red; font-weight:bold;'>{stress_label}</span>",
                    unsafe_allow_html=True
                )
                if pred_value > 8.5:
                    st.markdown(
                        "<span style='color:red; font-weight:bold;'>PLEASE SEEK PROFESSIONAL HELP.</span>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f"<span style='color:green; font-weight:bold;'>{stress_label}</span>",
                    unsafe_allow_html=True
                )