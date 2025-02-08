import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data():
    """Load and prepare the salary dataset"""
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    return dataset, X, y

def train_model(X, y):
    """Train the linear regression model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor, X_train, X_test, y_train, y_test

def create_prediction_plot(X_train, y_train, regressor):
    """Create the scatter plot with regression line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train, y_train, color='red', label='Actual Data')
    ax.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
    ax.set_title('Salary vs Experience')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    ax.legend()
    return fig

def main():
    st.set_page_config(page_title="Salary Predictor", layout="wide")
    
    st.title("💰 Salary Prediction App")
    st.write("Predict salary based on years of experience using Linear Regression")
    
    # Load data and train model
    try:
        dataset, X, y = load_data()
        regressor, X_train, X_test, y_train, y_test = train_model(X, y)
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Visualization")
            fig = create_prediction_plot(X_train, y_train, regressor)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Make a Prediction")
            
            # Add input for years of experience
            years_exp = st.number_input(
                "Enter years of experience:",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5
            )
            
            if st.button("Predict Salary"):
                prediction = regressor.predict([[years_exp]])[0]
                st.success(f"Predicted Salary: ${prediction:,.2f}")
        
        # Show dataset
        st.subheader("Dataset Preview")
        st.dataframe(dataset.head())
        
        # Model Performance Metrics
        st.subheader("Model Performance")
        train_score = regressor.score(X_train, y_train)
        test_score = regressor.score(X_test, y_test)
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Training Score (R²)", f"{train_score:.3f}")
        with metric_col2:
            st.metric("Testing Score (R²)", f"{test_score:.3f}")
        
    except FileNotFoundError:
        st.error("Please ensure 'Salary_Data.csv' is in the same directory as the app.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()