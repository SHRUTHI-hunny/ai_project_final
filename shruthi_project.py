# Adapted Streamlit Dashboard for Education & Career Success Dataset

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import base64
import io

st.set_page_config(page_title="Career Success Dashboard", layout="wide")

# Background styling
st.markdown("""
    <style>
    .stApp {
        background: url(data:image/png;base64,""" + base64.b64encode(open("bgimg.jpg", "rb").read()).decode() + """) no-repeat center center fixed;
        background-size: cover;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.6) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #fefce8 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #f1c40f;'>🎓 Career Success Prediction Dashboard</h1>
    <p style='text-align: center; color: #ecf0f1;'>Analyze, Visualize, and Predict Career Outcomes</p>
    <hr style='border: 1px solid #555;' />
""", unsafe_allow_html=True)
od=pd.read_csv("education_career_success_60_samples.csv")
@st.cache_data
def load_data():
    df = pd.read_csv("education_career_success_60_samples.csv")
    df.drop(columns=['ID', 'Current_Job_Title'], inplace=True, errors='ignore')
    df['career_success'] = df['Annual_Salary_USD'] > 90000
    df['gpa_experience_ratio'] = df['GPA'] / df['Years_of_Experience'].replace(0, 0.1)
    df['promotion_cert_ratio'] = df['Promotions_Received'] / df['Certifications_Count'].replace(0, 1)
    for col in ['Gender', 'Education_Level', 'Field_of_Study']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

df = load_data()
X = df.drop(['Annual_Salary_USD', 'career_success'], axis=1)
y = df['career_success']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

st.sidebar.title("📂 Sections")
section = st.sidebar.radio("Navigate", [
    "Data Overview", "Visualizations", "Model: Logistic Regression",
    "Model: Polynomial Regression", "Model: Random Forest", "KMeans Clustering"])

if section == "Data Overview":
    st.subheader("🧾 Data Preview")
    st.write(od.head(100))
    st.subheader('Data Description')
    st.dataframe(df.describe(include='all'))
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))

elif section == "Visualizations":
    st.subheader("📊 Visualization Section")
    viz_option = st.sidebar.selectbox("Choose a visualization:", (
        "Career Success Distribution", "Education Level by Success",
        "GPA vs Experience", "Feature Correlation Heatmap"))

    if viz_option == "Career Success Distribution":
        fig1 = px.pie(df, names='career_success', title='Career Success Distribution')
        st.plotly_chart(fig1, use_container_width=True)

    elif viz_option == "Education Level by Success":
        fig2 = px.histogram(df, x='Education_Level', color='career_success', title='Education Level by Career Success')
        st.plotly_chart(fig2, use_container_width=True)

    elif viz_option == "GPA vs Experience":
        fig3 = px.scatter(df, x='Years_of_Experience', y='GPA', color='career_success')
        st.plotly_chart(fig3, use_container_width=True)

    elif viz_option == "Feature Correlation Heatmap":
        fig4 = px.imshow(df.corr(numeric_only=True), text_auto=True, aspect="auto")
        st.plotly_chart(fig4, use_container_width=True)

elif section == "Model: Logistic Regression":
    st.subheader("📈 Logistic Regression")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=['Not Successful', 'Successful'], y=['Not Successful', 'Successful'])
    st.plotly_chart(fig_cm, use_container_width=True)

elif section == "Model: Polynomial Regression":
    st.subheader("🧮 Polynomial Logistic Regression")
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LogisticRegression(max_iter=1000)
    poly_model.fit(X_train_poly, y_train)
    y_poly_pred = poly_model.predict(X_test_poly)
    st.text(classification_report(y_test, y_poly_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_poly_pred):.2f}")
    cm_poly = confusion_matrix(y_test, y_poly_pred)
    fig_poly = px.imshow(cm_poly, text_auto=True, x=['Not Successful', 'Successful'], y=['Not Successful', 'Successful'])
    st.plotly_chart(fig_poly, use_container_width=True)

elif section == "Model: Random Forest":
    st.subheader("🌲 Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_rf_pred = rf.predict(X_test)
    st.text(classification_report(y_test, y_rf_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_rf_pred):.2f}")
    cm_rf = confusion_matrix(y_test, y_rf_pred)
    fig_rf = px.imshow(cm_rf, text_auto=True, x=['Not Successful', 'Successful'], y=['Not Successful', 'Successful'])
    st.plotly_chart(fig_rf, use_container_width=True)
    st.subheader("📊 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig_imp, use_container_width=True)

elif section == "KMeans Clustering":
    st.subheader("🤖 KMeans Clustering")
    from sklearn.cluster import KMeans
    from kneed import KneeLocator
    kmeans = KMeans(random_state=42)
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans.set_params(n_clusters=k)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    kn = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
    optimal_k = kn.elbow
    st.write(f"Optimal number of clusters: {optimal_k}")
    fig_kmeans = px.line(x=k_range, y=inertia, labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    fig_kmeans.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text="Optimal K")
    st.plotly_chart(fig_kmeans, use_container_width=True)

