%%writefile PyCaret2.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from pycaret.regression import *
from ydata_profiling import ProfileReport
from pycaret.regression import setup as set_reg, compare_models as compare_regression, evaluate_model as evaluate_reg, finalize_model as finalize_reg, predict_model as predict_reg
from pycaret.classification import setup as set_class, compare_models as compare_classification, evaluate_model as evaluate_class, finalize_model as finalize_class, predict_model as predict_class
from streamlit_pandas_profiling import st_profile_report

# Streamlit configuration
st.set_page_config(
    page_title="modeling app",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling Streamlit
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .stWarning {
            color: red;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to read a DataFrame from an uploaded file
def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.type == 'text/plain':
            df = pd.read_csv(uploaded_file, delimiter='\t')
        elif uploaded_file.type == 'application/json':
            df = pd.read_json(uploaded_file)
        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
        else:
            st.warning("Unsupported file format.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a file")

# If a file is uploaded, process it
if uploaded_file:
    df = read_uploaded_file(uploaded_file)

    if df is not None:
        # Display the first few rows of the DataFrame
        st.title("Uploaded Data Preview")
        st.write(df.head(10))

        # Generate and display a profile report when the button is clicked
        if st.button("Generate Data Overview"):
            profile_report = ProfileReport(df, title='Data Profile Report', explorative=True)
            st_profile_report(profile_report)

        # Fill all missing values with mean or median or mode
        st.title("Handle Missing Values for All Columns")
        st.write("note:the way you going to choose it 's on the whole column")
        global_fill_method = st.radio("Choose how to fill all missing values:", ['Fill with Mean', 'Fill with Median','Fill with Mode'])

        if global_fill_method == 'Fill with Mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if df[col].dtype != 'object']  # Exclude categorical columns
            df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))
            st.success("All numeric columns filled with mean.")
            st.write(df.head(10))

        elif global_fill_method == 'Fill with Median':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if df[col].dtype != 'object']  # Exclude categorical columns
            df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))
            st.success("All numeric columns filled with median.")
            st.write(df.head(10))

        elif global_fill_method == 'Fill with Mode':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if df[col].dtype != 'object']  # Exclude categorical columns
            df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mode()))
            st.success("All numeric columns filled with mode.")
            st.write(df.head(10))

        st.title("Delete a Specific Column")
        delete_column = st.multiselect("Select a column to delete", df.columns)
        if st.button(f"Delete '{delete_column}' column"):
            df.drop(columns=delete_column, inplace=True)
            st.success(f"The column '{delete_column}' has been deleted.")
            st.write(df.head(10))

        st.title("Data Encoding")
        # Encode Categorical Columns
        categorical_columns = st.radio("Choose how to handle categorical columns:", ['One-Hot Encoding', 'Label Encoding'])
        if categorical_columns == 'One-Hot Encoding':
            categorical_columns = df.select_dtypes(include=['object']).columns
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            st.write(df.head(10))
        elif categorical_columns == 'Label Encoding':
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = label_encoder.fit_transform(df[col])
      
      
        st.title("Visualization")
        st.write("Select the type of visualization to generate:")
        visualization_type = st.selectbox("Select Visualization Type", ["Histogram", "Correlation Heatmap"])

        if visualization_type == "Histogram":
            selected_column = st.selectbox("Select a column for the histogram", df.columns)
            plt.figure(figsize=(8, 6))
            sns.histplot(df[selected_column].dropna(), kde=True)
            plt.xlabel(selected_column)
            plt.ylabel("Frequency")
            st.pyplot()

        elif visualization_type == "Correlation Heatmap":
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot()

        # PyCaret Setup
        st.title("PyCaret Setup")
        target_column = st.selectbox("Select the target column", df.columns)
        target_column1 = str(df[target_column].dtype)
        if target_column1 in ['int64', 'float64']:
            regression = set_reg(df, target=target_column)
            best = compare_regression()
            finalize_reg(best)
            evaluate_reg(best)
            prediction = predict_reg(best, data=df)
        elif target_column1 == 'object':
            classification = set_class(df, target=target_column, fold_strategy='stratifiedkfold')
            best = compare_classification()
            finalize_class(best)
            evaluate_class(best)
            prediction = predict_class(best, data=df)

        st.write(best)
        st.title("Model Prediction")
        st.write(prediction)
        st.table(best)
