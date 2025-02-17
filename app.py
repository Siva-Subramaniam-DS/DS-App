import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Data
@st.cache_data
def load_data():
    google_df = pd.read_csv("googleplaystore.csv")
    apple_df = pd.read_csv("AppleStore.csv")

    # Strip whitespace from column names to prevent errors
    google_df.columns = google_df.columns.str.strip()
    apple_df.columns = apple_df.columns.str.strip()

    return google_df, apple_df

# Preprocess Data
def preprocess_data(google_df, apple_df):
    google_df = google_df[['App Name', 'Category', 'Rating', 'Reviews']].dropna()
    google_df['Reviews'] = google_df['Reviews'].astype(int)

    # Ensure 'AppName' exists in Apple DataFrame
    if 'AppName' not in apple_df.columns:
        st.error("Error: 'AppName' column not found in Apple DataFrame!")
        st.stop()

    apple_df = apple_df[['AppName', 'prime_genre', 'user_rating', 'rating_count_tot']].dropna()
    apple_df['rating_count_tot'] = apple_df['rating_count_tot'].astype(int)

    return google_df, apple_df

# Sentiment Labeling
def assign_sentiment(rating):
    if rating > 3.5:
        return "Positive"
    elif rating >= 2.5:
        return "Neutral"
    else:
        return "Negative"

# Train ML Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report

# Streamlit UI
st.set_page_config(page_title="App Store Sentiment Analysis", layout="wide")
st.title("📊 Sentiment Analysis of Google Play & Apple Store Apps")

# Load & Process Data
google_df, apple_df = load_data()
google_df, apple_df = preprocess_data(google_df, apple_df)

google_df["Sentiment"] = google_df["Rating"].apply(assign_sentiment)
apple_df["Sentiment"] = apple_df["user_rating"].apply(assign_sentiment)

# Sidebar Options
st.sidebar.header("Select Store & Category")
store_option = st.sidebar.radio("Choose Store:", ["Google Play Store", "Apple Store", "Both"])
category_option = st.sidebar.selectbox("Select Category", 
                                       google_df["Category"].unique() if store_option != "Apple Store" 
                                       else apple_df["prime_genre"].unique())

# Filter Data
if store_option == "Google Play Store":
    filtered_df = google_df[google_df["Category"] == category_option]
elif store_option == "Apple Store":
    filtered_df = apple_df[apple_df["prime_genre"] == category_option]
else:
    google_filtered = google_df[google_df["Category"] == category_option]
    apple_filtered = apple_df[apple_df["prime_genre"] == category_option]
    filtered_df = pd.concat([google_filtered, apple_filtered])

# Display Data
st.write(f"Showing data for: **{category_option}** ({store_option})")
st.dataframe(filtered_df)

# Sentiment Distribution Plot
fig, ax = plt.subplots(figsize=(15, 5))
sns.countplot(data=filtered_df, x="Sentiment", palette="coolwarm", ax=ax)
st.pyplot(fig)

# Train Model
st.subheader("Model Performance")
X = filtered_df[["Rating" if store_option == "Google Play Store" else "user_rating"]].values
y = filtered_df["Sentiment"]
model, accuracy, report = train_model(X, y)

st.write(f"**Accuracy:** {accuracy:.2f}")
st.write("**Classification Report:**")
st.json(report)

# Search Box for App Name
st.sidebar.header("🔍 Search for an App")
search_query = st.sidebar.text_input("Enter App Name:")

if search_query:
    if store_option == "Google Play Store":
        search_result = google_df[google_df["App Name"].str.contains(search_query, case=False, na=False)]
    elif store_option == "Apple Store":
        search_result = apple_df[apple_df["AppName"].str.contains(search_query, case=False, na=False)]
    else:
        google_search = google_df[google_df["App Name"].str.contains(search_query, case=False, na=False)]
        apple_search = apple_df[apple_df["AppName"].str.contains(search_query, case=False, na=False)]
        search_result = pd.concat([google_search, apple_search])

    if not search_result.empty:
        st.write(f"### Search Results for '{search_query}':")
        st.dataframe(search_result)

        sentiment_counts = search_result["Sentiment"].value_counts().to_dict()
        st.write(f"#### Sentiment Analysis for '{search_query}':")
        for sentiment, count in sentiment_counts.items():
            st.write(f"{sentiment}: {count} apps")

    else:
        st.write(f"No apps found with name '{search_query}'.")
