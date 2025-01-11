import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data (use your cleaned data file path here)
data = pd.read_csv("winequality-red.csv")

# Outlier removal as per your earlier code
from scipy import stats
Z = np.abs(stats.zscore(data))
data = data[(Z < 3).all(axis=1)]

# Splitting data into X (features) and Y (target)
X = data.drop(columns='quality')
Y = data['quality']

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, Y)

# Streamlit app
st.set_page_config(page_title="Red Wine Quality Predictor 🍷", layout="wide")

# Title and description
st.title("🍷 Red Wine Quality Prediction App 🍇")
st.markdown("""
Welcome to the Red Wine Quality Predictor! 
Enter the details of your wine below to predict its quality. 🥂
""")

# Sidebar
st.sidebar.title("About the App 📘")
st.sidebar.markdown("""
This app uses a **Random Forest Classifier** to predict wine quality based on several features.
### Features Considered:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
""")
st.sidebar.markdown("Made with ❤️ by Hardik Arora")

# User input form
st.header("Enter Wine Features 🍇")
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", min_value=4.0, max_value=16.0, value=7.4, step=0.1)
    volatile_acidity = st.slider("Volatile Acidity", min_value=0.1, max_value=1.6, value=0.7, step=0.01)
    citric_acid = st.slider("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    residual_sugar = st.slider("Residual Sugar", min_value=0.5, max_value=20.0, value=1.9, step=0.1)
    chlorides = st.slider("Chlorides", min_value=0.01, max_value=0.2, value=0.076, step=0.001)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", min_value=1.0, max_value=80.0, value=11.0, step=1.0)

with col2:
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", min_value=6.0, max_value=300.0, value=34.0, step=1.0)
    density = st.slider("Density", min_value=0.990, max_value=1.005, value=0.9978, step=0.0001)
    pH = st.slider("pH", min_value=2.8, max_value=4.0, value=3.51, step=0.01)
    sulphates = st.slider("Sulphates", min_value=0.3, max_value=2.0, value=0.56, step=0.01)
    alcohol = st.slider("Alcohol", min_value=8.0, max_value=15.0, value=9.4, step=0.1)

# Prediction
st.subheader("Wine Quality Prediction 🍷")
user_input = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# Prediction result
predicted_quality = rf_model.predict(user_input)[0]
st.markdown(f"### Predicted Quality: **{predicted_quality}** ⭐️")

# Visualizations
st.header("Correlation Heatmap 📊")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.success("Enjoy using the app! 🍷")

