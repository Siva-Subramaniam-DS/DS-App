import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Page Config
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Custom CSS for sections
st.markdown(
    """
    <style>
        .section {
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .start-section { background-color: #2E86C1; text-align: center; color: white; }
        .home-section { background-color: #F4D03F; }
        .analysis-section { background-color: #A2D9CE; }
        .model-section { background-color: #E59866; }
        .chat-section { background-color: #D7BDE2; }
        .about-section { background-color: #5D6D7E; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# **1. Start Section**
st.markdown('<div class="section start-section">', unsafe_allow_html=True)
st.header("🎬 Welcome to Sentiment Analysis App")
start_button = st.button("Start Analysis")
st.markdown('</div>', unsafe_allow_html=True)

if start_button:
    # **2. Home Section**
    st.markdown('<div class="section home-section">', unsafe_allow_html=True)
    st.header("🏠 Home")
    st.write("This app analyzes user sentiment for apps in the Google Play Store and Apple App Store.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # **3. App Analysis Section**
    st.markdown('<div class="section analysis-section">', unsafe_allow_html=True)
    st.header("📊 App Analysis")
    st.write("Here, you can explore app sentiment based on user reviews.")
    @st.cache_data
    def load_data():
        google_df = pd.read_csv("googleplaystore.csv")
        apple_df = pd.read_csv("AppleStore.csv")

        google_df.columns = google_df.columns.str.strip()
        apple_df.columns = apple_df.columns.str.strip()

        return google_df, apple_df
    st.markdown('</div>', unsafe_allow_html=True)
    
    # **4. Model Performance Section**
    st.markdown('<div class="section model-section">', unsafe_allow_html=True)
    st.header("🤖 Model Performance")
    st.write("Detailed model evaluation and performance metrics.")
    # Preprocess Data
    def preprocess_data(google_df, apple_df):
        google_df = google_df[['App Name', 'Category', 'Rating', 'Reviews']].dropna()
        google_df['Reviews'] = google_df['Reviews'].astype(int)
        
        if 'AppName' not in apple_df.columns:
            st.error("Error: 'AppName' column not found in Apple DataFrame!")
            st.stop()

        apple_df = apple_df[['AppName', 'prime_genre', 'user_rating', 'rating_count_tot']].dropna()
        apple_df['rating_count_tot'] = apple_df['rating_count_tot'].astype(int)
        
        return google_df, apple_df

    # Sentiment Analysis
    def assign_sentiment(rating):
        if rating > 3.5:
            return "Positive"
        elif rating >= 2.5:
            return "Neutral"
        else:
            return "Negative"

    # Train Model
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return model, accuracy, report
    st.markdown('</div>', unsafe_allow_html=True)
    
    # **5. Lexi Chat Section**
    st.markdown('<div class="section chat-section">', unsafe_allow_html=True)
    st.header("💬 Lexi Chat")
    st.write("Chat with Lexi AI for insights and queries.")
    # Placeholder for chatbot content
    st.markdown('</div>', unsafe_allow_html=True)
    
    # **6. About Us Section**
    st.markdown('<div class="section about-section">', unsafe_allow_html=True)
    st.header("👥 About Us")
    st.write("Learn more about the team behind this project.")
    st.markdown('</div>', unsafe_allow_html=True)
