import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Page Config & Styling
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Custom CSS for styling sections
st.markdown(
    """
    <style>
        .section {
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
        .tab1-section { background-color: #2E86C1; } /* Blue */
        .tab2-section { background-color: #F4D03F; color: black; } /* Yellow */
        .tab3-section { background-color: #A2D9CE; color: black; } /* Green */
    </style>
    """,
    unsafe_allow_html=True
)


# Load Data
@st.cache_data
def load_data():
    google_df = pd.read_csv("googleplaystore.csv")
    apple_df = pd.read_csv("AppleStore.csv")

    google_df.columns = google_df.columns.str.strip()
    apple_df.columns = apple_df.columns.str.strip()

    return google_df, apple_df

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

# Load & Process Data
google_df, apple_df = load_data()
google_df, apple_df = preprocess_data(google_df, apple_df)

google_df["Sentiment"] = google_df["Rating"].apply(assign_sentiment)
apple_df["Sentiment"] = apple_df["user_rating"].apply(assign_sentiment)

# Sidebar
st.sidebar.image("Lexi.jpg", width=180)
st.sidebar.header("📊 Select Store & Category")
store_option = st.sidebar.radio("Choose Store:", ["Google Play Store", "Apple Store", "Both"])
category_option = st.sidebar.selectbox("Select Category", 
                                       google_df["Category"].unique() if store_option != "Apple Store" 
                                       else apple_df["prime_genre"].unique())

# Search Feature
st.sidebar.header("🔍 Search for an App")
search_query = st.sidebar.text_input("Enter App Name:")



# Filter Data
if store_option == "Google Play Store":
    filtered_df = google_df[google_df["Category"] == category_option]
elif store_option == "Apple Store":
    filtered_df = apple_df[apple_df["prime_genre"] == category_option]
else:
    google_filtered = google_df[google_df["Category"] == category_option]
    apple_filtered = apple_df[apple_df["prime_genre"] == category_option]
    filtered_df = pd.concat([google_filtered, apple_filtered])
    
    
    

# Tabs for Sections
tab1, tab2, tab3 = st.tabs(["📈 App Analysis", "🤖 ML Model", "💬 About us"])

with tab1:
    st.subheader(f"Showing data for: **{category_option}** ({store_option})")
    st.dataframe(filtered_df)
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.countplot(data=filtered_df, x="Sentiment", palette="coolwarm", ax=ax)
    st.pyplot(fig)
    
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

with tab2:
    st.subheader("Machine Learning Model Performance")

    # Extract features and target variable based on store selection
    X = filtered_df[["Rating" if store_option == "Google Play Store" else "user_rating"]].values
    y = filtered_df["Sentiment"]

    # Train the model
    model, accuracy, report = train_model(X, y)

    # Display accuracy
    st.write(f"### **Model Accuracy: {accuracy:.2f}**")

    # Convert classification report to DataFrame for table display
    report_df = pd.DataFrame(report).transpose()

    # Separate sentiment results
    st.write("### **Sentiment Classification Report**")
    st.dataframe(report_df)

    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    labels = sorted(set(y))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    # Display confusion matrix
    st.write("### **Confusion Matrix**")
    st.dataframe(conf_matrix_df)

    # Additional Analysis
    st.write("### **Additional Insights**")
    sentiment_counts = filtered_df["Sentiment"].value_counts().to_frame(name="Count")
    st.dataframe(sentiment_counts)

    # Show feature importance
    feature_importance = model.feature_importances_
    feature_df = pd.DataFrame(
        {"Feature": ["Rating" if store_option == "Google Play Store" else "user_rating"], 
         "Importance": feature_importance}
    )
    
    st.write("### **Feature Importance**")
    st.dataframe(feature_df)




with tab3:
    
    st.markdown(
        """
        ## **About Us**  
        
        Welcome to the **Sentiment Analysis App**, a platform designed to analyze app reviews and provide sentiment insights based on user feedback.
        
        ### **What is Machine Learning?**  
        Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data and make predictions without being explicitly programmed. In this app, ML helps classify app reviews into Positive, Neutral, or Negative sentiments.  

        ### **What is Model Performance?**  
        Model performance refers to how well a machine learning model makes predictions. It is measured using various metrics like accuracy, precision, recall, and F1-score.  

        ### **What is Model Accuracy?**  
        Accuracy is a metric that measures how often the model's predictions are correct. It is calculated as:  
        \[
        Accuracy = \frac{Correct Predictions}{Total Predictions}
        \]
        A high accuracy indicates that the model is performing well, but it should be evaluated alongside other metrics.  

        ### **Sentiment Classification Report**  
        The classification report provides an analysis of the model’s ability to classify sentiments correctly. It includes:  

        - **Precision**: The proportion of true positive predictions among all predicted positives.  
        - **Recall**: The proportion of true positives correctly identified out of all actual positives.  
        - **F1-score**: The harmonic mean of precision and recall, balancing the two metrics.  
        - **Support**: The number of actual instances in each class.  

        ### **Confusion Matrix**  
        A confusion matrix is a table used to evaluate the performance of a classification model. It shows:  

        - **True Positives (TP)**: Correctly predicted positive cases.  
        - **True Negatives (TN)**: Correctly predicted negative cases.  
        - **False Positives (FP)**: Incorrectly predicted positive cases (Type I Error).  
        - **False Negatives (FN)**: Incorrectly predicted negative cases (Type II Error).  

        ### **Feature Importance**  
        Feature importance helps understand which input variables (features) contribute the most to the model’s decision-making process. In this app, features like **Rating**, **Reviews**, and **User Feedback** play a crucial role in predicting sentiment.  

        ### **Special Thanks**  
        This project would not have been possible without the guidance, ideas, and support from **Karthick Kumar R ( M.Tech )**. His insights and contributions played a crucial role in shaping this application.  

        ### **Developed by**  
        **Siva Subramanian R ( M.Tech )** - Passionate about AI, Data Science, and Web Development. Always looking to create impactful projects that make a difference.  

        ### **Technology Stack**  
        This app is built using:  
        ✅ **Python** - The backbone of data processing and machine learning.  
        ✅ **Streamlit** - For an interactive and visually appealing web interface.  
        ✅ **Pandas & NumPy** - Data processing and manipulation.  
        ✅ **Matplotlib & Seaborn** - For powerful visualizations.  
        ✅ **Scikit-Learn** - To build and train machine learning models.  
        
        ### **Our Vision**  **Our mission**
        - Our mission is to empower developers, businesses, and users with real-time sentiment analysis, helping them make data-driven decisions.  
        - To revolutionize the way app reviews are analyzed and interpreted.
        - To deliver a user-friendly and interactive experience for analyzing app sentiment.  
        - To enhance app insights using AI and Machine Learning.  
        - To provide valuable feedback for app developers to improve user satisfaction.

        ### **Copyright & License**  
        © 2025 Sentiment App Analysis | Powered by AI & Streamlit  
        All rights reserved. Unauthorized reproduction or distribution of this application or its content is prohibited.  
        """,
        unsafe_allow_html=True
    )


# Footer
st.markdown(
    """
    <hr style="border:1px solid white">
    <p style="text-align:center;"> © 2025 Sentiment App Analysis | Powered by AI & Streamlit </p>
    """,
    unsafe_allow_html=True
)
