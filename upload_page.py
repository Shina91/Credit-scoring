import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
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
    stress_label = 'stress' if pred_value > 0.5 else 'no-stress'

    return pred_value, sentiment_label, stress_label

def show_upload_page():
    st.title("STRESS DETECTION AND SENTIMENT ANALYSIS ")
    st.image('stress image.png', caption='Stress Analysis', use_column_width=True)


    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'text' in df.columns:
            # Initialize session state variables
            if "data_preview" not in st.session_state:
                st.session_state.data_preview = False
            if "data_summary" not in st.session_state:
                st.session_state.data_summary = False
            if "word_cloud" not in st.session_state:
                st.session_state.word_cloud = False
            if "analyze_text" not in st.session_state:
                st.session_state.analyze_text = False
                st.session_state.pred_value = None
                st.session_state.sentiment_label = None
                st.session_state.stress_label = None

            # Button for Data Preview with a unique key
            if st.button("Show Data Preview", key="data_preview_btn"):
                st.session_state.data_preview = True

            if st.session_state.data_preview:
                st.subheader("Dataset Preview")
                st.write(df.head())

            # Button for Data Summary with a unique key
            if st.button("Show Data Summary", key="data_summary_btn"):
                st.session_state.data_summary = True

            if st.session_state.data_summary:
                st.subheader("Data Summary")
                st.write(df.describe())

            # Button for Word Cloud with a unique key
            if st.button("Generate Word Cloud", key="word_cloud_btn"):
                st.session_state.word_cloud = True

            if st.session_state.word_cloud:
                st.subheader("Word Cloud")
                text = " ".join(str(i) for i in df.text)
                wc = WordCloud(background_color="white", width=1200, height=600,
                               contour_width=0, contour_color="#410F01", max_words=1000,
                               scale=1, collocations=False, repeat=True, min_font_size=1).generate(text)
                
                plt.figure(figsize=[15, 7])
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)

            text_to_analyze = st.selectbox("Select a text to analyze", df['text'])

            # A button to trigger text analysis
            if st.button("Analyze Text", key="analyze_text_btn"):
                st.session_state.analyze_text = True
                st.session_state.pred_value, st.session_state.sentiment_label, st.session_state.stress_label = analyze_text(text_to_analyze)

            if st.session_state.analyze_text:
                st.write(f"**Selected Text:** {text_to_analyze}")
                st.write(f"The Prediction Value: {st.session_state.pred_value:.2f}")
                st.write(f"The Predicted Sentiment: {st.session_state.sentiment_label}")

                # Display the predicted stress level with color coding
                if st.session_state.stress_label == 'STRESS':
                    st.markdown(
                        f"**The Predicted Stress Level:** <span style='color:red; font-weight:bold;'>{st.session_state.stress_label}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"**The Predicted Stress Level:** <span style='color:green; font-weight:bold;'>{st.session_state.stress_label}</span>",
                        unsafe_allow_html=True
                    )

                if st.session_state.pred_value > 8.5:
                    st.markdown("<span style='color:red; font-weight:bold;'>Please consider seeing a mental health professional.</span>",
                        unsafe_allow_html=True
                    )

        else:
            st.error("The uploaded CSV file does not contain a 'text' column.")
