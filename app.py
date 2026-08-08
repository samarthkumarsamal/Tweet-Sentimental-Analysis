import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Tweet Sentiment Analyzer | AI-Powered Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# -----------------------------------------------------

# ---------------- Prediction Function ----------------
def prediction(model, text):
    """
    Predict sentiment of the given text using the loaded model
    """
    pred = model.predict([text])[0]
    
    # Check if model has predict_proba method (SVC might not have it by default)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([text])[0]
        negative_prob = proba[0]
        positive_prob = proba[1]
    else:
        # For SVC without probability, use decision_function
        if hasattr(model, 'decision_function'):
            decision = model.decision_function([text])[0]
            # Convert decision function to probability-like scores
            # Using sigmoid function to normalize
            import math
            prob = 1 / (1 + math.exp(-decision))
            positive_prob = prob
            negative_prob = 1 - prob
        else:
            # Fallback: assign 100% confidence to predicted class
            if pred == 1:
                positive_prob = 1.0
                negative_prob = 0.0
            else:
                positive_prob = 0.0
                negative_prob = 1.0
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = max(positive_prob, negative_prob)
    
    return {
        'Predicted_Label': pred,
        'Sentiment': sentiment,
        'Confidence': confidence,
        'Negative': negative_prob,
        'Positive': positive_prob
    }
# -----------------------------------------------------

# ---------------- Enhanced CSS Styles ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #8b5cf6;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --dark: #0f172a;
    --dark-light: #1e293b;
    --gray: #64748b;
    --light: #f8fafc;
}

* {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}

/* Header Section */
.hero-section {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 24px;
    padding: 60px 40px;
    margin-bottom: 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.hero-title {
    font-size: 56px;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 24px;
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: 18px;
    color: #94a3b8;
    font-weight: 400;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.8;
    position: relative;
    z-index: 1;
    text-align: center;
    overflow: hidden;
    border-right: 3px solid #6366f1;
    white-space: nowrap;
    display: inline-block;
    animation: typing 4s steps(100, end), blink-caret 0.75s step-end infinite;
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink-caret {
    from, to { border-color: transparent; }
    50% { border-color: #6366f1; }
}

/* Section Headers */
.section-container {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 32px;
    backdrop-filter: blur(10px);
}

.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.section-description {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 24px;
    line-height: 1.6;
}

/* Input Styling */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 2px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 16px !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

.char-counter {
    display: flex;
    justify-content: space-between;
    margin-top: 12px;
    padding: 12px 16px;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 8px;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
}

.counter-item {
    color: #94a3b8;
}

.counter-value {
    color: #6366f1;
    font-weight: 600;
}

/* Model Selection */
.model-card {
    background: rgba(15, 23, 42, 0.6);
    border: 2px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
    cursor: pointer;
    margin-bottom: 12px;
}

.model-card:hover {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(99, 102, 241, 0.2);
}

.model-card.selected {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.15);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
}

.model-name {
    font-size: 15px;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 4px;
}

.model-description {
    font-size: 13px;
    color: #94a3b8;
}

/* Radio Buttons */
div[role="radiogroup"] {
    gap: 12px !important;
}

div[role="radiogroup"] label {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 2px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

div[role="radiogroup"] label:hover {
    border-color: #6366f1 !important;
    background: rgba(99, 102, 241, 0.1) !important;
    transform: translateX(4px) !important;
}

div[role="radiogroup"] label:has(input:checked) {
    border-color: #6366f1 !important;
    background: rgba(99, 102, 241, 0.15) !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Results Section */
.results-container {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 32px;
    margin-top: 32px;
}

.result-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    height: 100%;
    transition: all 0.3s ease;
}

.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2);
}

.result-label {
    font-size: 14px;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
}

.result-value {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-value.positive {
    color: #10b981;
}

.result-value.negative {
    color: #ef4444;
}

.result-icon {
    font-size: 32px;
    margin-bottom: 12px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 24px;
}

.stat-item {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-item:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    border-color: #6366f1;
}

.stat-label {
    font-size: 13px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    font-weight: 600;
}

.stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
}

/* Model Info Badge */
.model-info-badge {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    text-align: center;
}

.model-info-label {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.model-info-name {
    font-size: 16px;
    color: #a5b4fc;
    font-weight: 600;
}

/* Chart Headers */
.chart-header {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
    text-align: center;
    margin: 32px 0 16px 0;
}

/* Footer */
.footer {
    margin-top: 80px;
    padding: 40px;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    text-align: center;
}

.footer-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
}

.footer-avatar {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 3px solid #6366f1;
    box-shadow: 0 0 24px rgba(99, 102, 241, 0.4);
    object-fit: cover;
}

.footer-text {
    text-align: left;
}

.footer-name {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 4px 0;
}

.footer-tagline {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeInUp 0.6s ease-out;
}

/* Spinner */
.stSpinner > div {
    border-color: #6366f1 transparent transparent transparent !important;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------

# ---------------- Hero Section ----------------
st.markdown("""
<div class="hero-section fade-in">
    <div style="display: flex; align-items: center; justify-content: center; gap: 30px; margin-bottom: 30px;">
        <div style="font-size: 64px; animation: float 3s ease-in-out infinite;">
            🐦
        </div>
        <div style="font-size: 72px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 20px; padding: 15px 25px; box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4); animation: pulse-glow 2s ease-in-out infinite;">
            💬
        </div>
        <div style="font-size: 64px; animation: float 3s ease-in-out infinite 0.5s;">
            📊
        </div>
    </div>
    <h1 class="hero-title">Tweet Sentiment Analyzer</h1>
    <div style="text-align: center;">
        <p class="hero-subtitle">
            Advanced machine learning platform for real-time sentiment classification. Leverage state-of-the-art NLP models to analyze Twitter content with precision and accuracy.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    image = Image.open("assets/tweetanalyze.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True)
except FileNotFoundError:
    pass
# -----------------------------------------------------

# ---------------- Input Section ----------------
st.markdown("""
<div class="section-container fade-in">
    <div class="section-header">
        <span>📝</span>
        <span>Input Configuration</span>
    </div>
    <div class="section-description">
        Enter the tweet which you want to analyze.
    </div>
</div>
""", unsafe_allow_html=True)

tweet_input = st.text_area(
    label="Tweet Content",
    placeholder="Enter tweet text for sentiment analysis...",
    height=150,
    label_visibility="collapsed"
)

if tweet_input:
    word_count = len(tweet_input.split())
    char_count = len(tweet_input)
    st.markdown(f"""
    <div class="char-counter">
        <div class="counter-item">
            <span>Characters:</span> <span class="counter-value">{char_count}</span>
        </div>
        <div class="counter-item">
            <span>Words:</span> <span class="counter-value">{word_count}</span>
        </div>
        <div class="counter-item">
            <span>Status:</span> <span class="counter-value">Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
# -----------------------------------------------------

# ---------------- Model Selection ----------------
st.markdown("""
<div class="section-container fade-in">
    <div class="section-header">
        <span>🤖</span>
        <span>Model Selection</span>
    </div>
    <div class="section-description">
        Choose from our suite of trained machine learning models. Each model employs different vectorization techniques and algorithms.
    </div>
</div>
""", unsafe_allow_html=True)

model_options = {
    "Model 1": "Models/lor-tfidf.joblib",
    "Model 2": "Models/lor-count.joblib",
    "Model 3": "Models/nb-tfidf.joblib",
    "Model 4": "Models/nb-count.joblib",
    "Model 5": "Models/svc-tfidf.joblib",
    "Model 6": "Models/svc-count.joblib"
}

model_names = {
    "Model 1": "Logistic Regression with TF-IDF Vectorization",
    "Model 2": "Logistic Regression with Count Vectorization",
    "Model 3": "Naive Bayes with TF-IDF Vectorization",
    "Model 4": "Naive Bayes with Count Vectorization",
    "Model 5": "Support Vector Classifier with TF-IDF Vectorization",
    "Model 6": "Support Vector Classifier with Count Vectorization"
}

model_descriptions = {
    "Model 1": "Statistical classification using term frequency-inverse document frequency weighting",
    "Model 2": "Linear model with bag-of-words feature representation",
    "Model 3": "Probabilistic classifier with TF-IDF feature extraction",
    "Model 4": "Bayesian approach with raw word count features",
    "Model 5": "Support vector machine with TF-IDF feature extraction",
    "Model 6": "SVM classifier with raw word count features"
}

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Model 1"

col1, col2 = st.columns([3, 1])

with col1:
    selected_model = st.radio(
        label="Select Model",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.selected_model),
        format_func=lambda x: f"{x}: {model_names[x]}",
        label_visibility="collapsed"
    )
    st.session_state.selected_model = selected_model

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🎯 Analyze Sentiment", use_container_width=True)
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.selected_model = "Model 1"
        st.rerun()
# -----------------------------------------------------

# ---------------- Prediction & Results ----------------
if predict_clicked:
    if not tweet_input.strip():
        st.warning("⚠️ Please enter tweet content before analysis.")
    else:
        model_path = model_options[selected_model]
        try:
            with st.spinner('🔮 Processing sentiment analysis...'):
                loaded_model = joblib.load(model_path)
                result = prediction(loaded_model, tweet_input)

            st.markdown(f"""
            <div class="model-info-badge">
                <div class="model-info-label">Active Model</div>
                <div class="model-info-name">{model_names[selected_model]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header" style="justify-content: center; margin: 32px 0;">
                <span>📊</span>
                <span>Analysis Results</span>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1], gap="large")

            with col1:
                sentiment_class = "positive" if result['Predicted_Label'] == 1 else "negative"
                icon = "✅" if result['Predicted_Label'] == 1 else "❌"
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Sentiment Classification</div>
                    <div class="result-icon">{icon}</div>
                    <div class="result-value {sentiment_class}">{result['Sentiment']}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Confidence Level</div>
                    <div class="result-value" style="color: #6366f1;">{result['Confidence']:.1%}</div>
                    <div style="margin-top: 16px; font-size: 13px; color: #94a3b8;">
                        Model certainty in prediction
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                fig_pie = go.Figure(go.Pie(
                    labels=["Negative", "Positive"],
                    values=[result['Negative'], result['Positive']],
                    hole=0.65,
                    marker_colors=["#ef4444", "#10b981"],
                    textinfo="none",
                    hovertemplate="<b>%{label}</b><br>Probability: %{value:.2%}<extra></extra>"
                ))
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    margin=dict(l=0,r=0,t=0,b=0),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=12)
                    ),
                    height=250
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(f"""
            <div class="stats-grid" style="margin-top: 32px;">
                <div class="stat-item">
                    <div class="stat-label">Negative Probability</div>
                    <div class="stat-value" style="color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);">{result['Negative']:.1%}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Positive Probability</div>
                    <div class="stat-value" style="color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);">{result['Positive']:.1%}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Prediction Strength</div>
                    <div class="stat-value" style="color: #6366f1; text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);">
                        {"High" if result['Confidence'] > 0.8 else "Medium" if result['Confidence'] > 0.6 else "Low"}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='chart-header'>Confidence Gauge</div>", unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['Confidence']*100,
                number={'suffix': "%", 'font': {'size': 48, 'color': '#f1f5f9'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                    'bar': {'color': "#10b981" if result['Predicted_Label']==1 else "#ef4444"},
                    'bgcolor': "rgba(15, 23, 42, 0.8)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(99, 102, 241, 0.3)",
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [60, 80], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                margin=dict(t=40,b=40,l=60,r=60),
                height=350
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("<div class='chart-header'>Probability Distribution</div>", unsafe_allow_html=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=['Negative', 'Positive'],
                y=[result['Negative']*100, result['Positive']*100],
                marker_color=['#ef4444', '#10b981'],
                text=[f"{result['Negative']:.1%}", f"{result['Positive']:.1%}"],
                textposition='outside',
                textfont=dict(size=16, color='#f1f5f9'),
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>"
            ))
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                yaxis={'title': 'Probability (%)', 'gridcolor': 'rgba(99, 102, 241, 0.2)', 'range': [0, 110]},
                xaxis={'title': ''},
                margin=dict(t=30,b=50,l=60,r=40),
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except FileNotFoundError:
            st.error(f"❌ Model file not found: {model_path}")
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
# -----------------------------------------------------
