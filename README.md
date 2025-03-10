# Sentiment Analysis App

## 📌 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Introduction
The **Sentiment Analysis App** is a web application built using **Streamlit** that analyzes user reviews of apps from **Google Play Store** and **Apple App Store**. It applies **machine learning models** to classify sentiments and provides insights into user opinions.

## 🚀 Features
- Supports **Google Play Store** and **Apple App Store** data.
- Sentiment classification using **Random Forest Classifier**.
- Interactive **data visualization** with Seaborn and Matplotlib.
- **Search functionality** to find app sentiments.
- **Lexi Chatbot** powered by an **LLM (Llava)** for user interaction.

## ⚙ Installation
```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📌 Usage
1. Select **Google Play Store**, **Apple Store**, or **Both** from the sidebar.
2. Choose an **App Category** to filter the results.
3. Search for a specific app to view its sentiment analysis.
4. View **sentiment distribution** via interactive charts.
5. Access **ML model performance** in the dedicated tab.
6. Chat with **Lexi Chatbot** for insights.

## 📊 Dataset
- **Google Play Store Dataset**: `googleplaystore.csv`
- **Apple App Store Dataset**: `AppleStore.csv`
- Both datasets contain app names, ratings, reviews, and categories.

## 🤖 Model Training
- **Algorithm Used**: Random Forest Classifier
- **Sentiment Labels**:
  - **Positive**: Rating > 3.5
  - **Neutral**: Rating between 2.5 - 3.5
  - **Negative**: Rating < 2.5

## 🛠 Technologies Used
- **Python**
- **Streamlit** (Frontend UI)
- **Pandas** (Data Processing)
- **Matplotlib & Seaborn** (Visualization)
- **Scikit-Learn** (ML Model)
- **TextBlob & VaderSentiment** (NLP Sentiment Analysis)
- **LangChain & OllamaLLM** (Chatbot)

## 📈 Results
- Achieved an **accuracy of ~85%** with Random Forest Classifier.
- Visual insights on sentiment distribution across app categories.

## 🔮 Future Enhancements
- Add **deep learning models** for improved sentiment analysis.
- Integrate **real-time review fetching** from app stores.
- Deploy chatbot using a **better NLP model**.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests.

## 📜 License
This project is licensed under the **MIT License**.

---

## 👨‍💻 Author  
- **Your Name**  
- GitHub: [@yourusername](https://github.com/Siva-Subramaniam-DS)  
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/r-siva-subramanaiam/)  

---
