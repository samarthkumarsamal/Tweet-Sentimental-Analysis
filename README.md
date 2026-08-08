# 🐦 Tweet Sentiment Analysis Web Application

## 🚀 Live Demo

🔗 **Try the Application Here:** https://tweet-sentimental-analysis-sam.streamlit.app/

Analyze public sentiment in real time using machine learning–powered tweet classification.

---

Dataset Link : https://www.kaggle.com/datasets/kazanova/sentiment140

## 📌 Project Overview

The **Tweet Sentiment Analysis Web Application** is an interactive machine learning platform that classifies user-input tweets as **Positive** or **Negative**. The application allows users to select from **six different trained machine learning models** and instantly view sentiment predictions along with supporting visualizations.

Built with **Streamlit**, the app combines robust natural language processing (NLP) techniques using **NLTK** with multiple machine learning algorithms developed using **scikit-learn**. It also provides insightful plots to help users understand sentiment distributions and model behavior.

This project is well-suited for NLP learners, data scientists, analysts, and developers interested in sentiment analysis applications.

---

## 🛠️ Tech Stack & Tools

| Technology        | Purpose                                              |
|------------------|------------------------------------------------------|
| 🐍 Python 3.7+    | Core programming language                            |
| 🐼 Pandas         | Data manipulation and preprocessing                  |
| 🔢 NumPy          | Numerical computations                               |
| 🧠 NLTK           | Text preprocessing and NLP utilities                 |
| 🤖 scikit-learn   | Machine learning models and evaluation               |
| 💾 Joblib         | Model serialization and loading                      |
| 🚀 Streamlit      | Interactive web application framework                |
| 📊 Plotly         | Interactive visualizations                           |
| 🎨 Matplotlib     | Static plots and visual analysis                     |
| 🌈 Seaborn        | Statistical data visualization                       |

---

## ✨ Key Features

- ✍️ **User Input Tweets**  
  Enter any custom tweet or short text for sentiment analysis.

- 🤖 **Multiple Model Selection**  
  Choose from **six different machine learning models** to compare predictions.

- 🔮 **Sentiment Prediction**  
  Classifies tweets as **Positive** or **Negative** in real time.

- 📊 **Visual Analytics**  
  View supporting plots such as:
  - Sentiment distribution
  - Model comparison visuals
  - Prediction confidence and insights

- ⚡ **Fast Inference**  
  Pre-trained models loaded efficiently using `joblib`.

- 📱 **Responsive UI**  
  Clean and user-friendly interface accessible on all devices.

---

## ⚙️ Setup Instructions (Local Development)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Tweet-Sentiment-Analysis.git
cd Tweet-Sentiment-Analysis
```

### 2. Create a Virtual Environment

``` bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows

``` bash
venv\Scripts\activate
```

#### Linux/macOS

``` bash
source venv/bin/activate
```

### 4. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 5. Run the Application

``` bash
streamlit run app.py
```

## 📁 Repository Structure

```plaintext
.
├── assets/
│   └── tweetanalyze.png              # Image used in the web application
│
├── Dataset/
│   ├── Tweet-Dataset.csv             # Training dataset
│   └── Testing-Dataset.csv           # Testing dataset
│
├── Models/
│   ├── lor-count.joblib              # Logistic Regression (Count Vectorizer)
│   ├── lor-tfidf.joblib              # Logistic Regression (TF-IDF)
│   ├── nb-count.joblib               # Naive Bayes (Count Vectorizer)
│   ├── nb-tfidf.joblib               # Naive Bayes (TF-IDF)
│   ├── svc-count.joblib              # Support Vector Classifier (Count Vectorizer)
│   └── svc-tfidf.joblib              # Support Vector Classifier (TF-IDF)
│
├── Notebooks/
│   ├── Logistic Regression/
│   │   ├── LOR_Count_Training.ipynb
│   │   ├── LOR_Count_Testing.ipynb
│   │   ├── LOR_TFIDF_Training.ipynb
│   │   └── LOR_TFIDF_Testing.ipynb
│   │
│   ├── Naive Bayes/
│   │   ├── NB_Count_Training.ipynb
│   │   ├── NB_Count_Testing.ipynb
│   │   ├── NB_TFIDF_Training.ipynb
│   │   └── NB_TFIDF_Testing.ipynb
│   │
│   ├── Support Vector Classifier/
│   │   ├── SVC_Count_Training.ipynb
│   │   ├── SVC_Count_Testing.ipynb
│   │   ├── SVC_TFIDF_Training.ipynb
│   │   └── SVC_TFIDF_Testing.ipynb
│   │
│   ├── Testing.ipynb                 # Manual tweet testing
│   └── EDA.ipynb                     # Exploratory Data Analysis
│
├── app.py                            # Streamlit application
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Usage Instructions

1. Launch the Streamlit application.
2. Enter a tweet or short text in the input field.
3. Select one of the six available machine learning models.
4. Click Analyze Sentiment.
5. View the predicted sentiment (Positive / Negative).
6. Explore additional plots and visual insights generated by the model.

---

## 👨‍💻 Contributing

### Contributions are welcome.

1. Fork the repository
2. Create a new feature branch:
``` bash
git checkout -b feature-name
```
3. Commit your changes:
``` bash
git commit -m "Add feature description"
```
4. Push to your branch:
``` bash
git push origin feature-name
```
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)**.

---

## 👤 Author

**Samarth Kumar Samal**
🔗 [GitHub Profile](https://github.com/Samarth-Kumar-Samal-Sam)

---

## 🙏 Acknowledgements

Special thanks to these fantastic tools and libraries:

* [NumPy](https://numpy.org/doc/stable/)
* [Pandas](https://pandas.pydata.org/docs/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [NLTK](https://www.nltk.org/)
* [Joblib](https://joblib.readthedocs.io/)
* [Streamlit](https://docs.streamlit.io/)
* [Plotly](https://plotly.com/python/)
---
