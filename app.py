import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from model import ModelTrainer
from utils import generate_insights, plot_feature_importance, calculate_contingency
import os
import io

# Error handling decorator
def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

# Set page title and configuration
st.set_page_config(
    page_title="Construction Cost Overrun Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Immediately show an instruction
st.info("👈 Please use the sidebar to upload your data or select sample data to begin.")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# App title and description
st.title("Nigerian Construction Project Cost Overrun Predictor")
st.markdown("""
This application helps predict cost overruns in Nigerian construction projects and recommends appropriate contingency allocations.
Upload your project data to get started or use the sample data for demonstration.
""")

# Sidebar for data upload and model parameters
with st.sidebar:
    st.header("Data Input")
    upload_option = st.radio(
        "Choose data source:",
        ("Upload CSV", "Use Sample Data")
    )
    
    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your project data CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        try:
            # Load sample data
            data = pd.read_csv("sample_data.csv")
            st.session_state.data = data
            st.success("Sample data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    st.header("Model Parameters")
    model_type = st.selectbox(
        "Select Model Type",
        ("Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regression")
    )
    
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    # Add button to trigger model training
    train_model = st.button("Train Model")

# Main content area
if st.session_state.data is not None:
    st.header("Data Overview")
    
    # Display data overview
    tab1, tab2 = st.tabs(["Data Preview", "Statistics"])
    
    with tab1:
        st.dataframe(st.session_state.data.head(10))
        st.write(f"Shape of data: {st.session_state.data.shape}")
    
    with tab2:
        st.write("Descriptive Statistics:")
        st.dataframe(st.session_state.data.describe())
    
    # Process data and train model when button is clicked
    if train_model:
        with st.spinner("Processing data and training model..."):
            # Process data
            processor = DataProcessor(st.session_state.data)
            X_train, X_test, y_train, y_test, feature_names = processor.preprocess_data(test_size=test_size)
            st.session_state.processor = processor
            
            # Train model
            model_trainer = ModelTrainer(model_type)
            model_trainer.train(X_train, y_train)
            
            # Make predictions and evaluate
            y_pred_train = model_trainer.predict(X_train)
            y_pred_test = model_trainer.predict(X_test)
            metrics = model_trainer.evaluate(y_test, y_pred_test)
            
            # Store results in session state
            st.session_state.model = model_trainer
            st.session_state.predictions = {
                'train': {'actual': y_train, 'predicted': y_pred_train},
                'test': {'actual': y_test, 'predicted': y_pred_test}
            }
            st.session_state.metrics = metrics
            st.session_state.feature_importance = model_trainer.get_feature_importance(feature_names)
            
            st.success("Model trained successfully!")

    # Display results if model has been trained
    if st.session_state.model is not None:
        st.header("Model Performance")
        
        # Display metrics
        metrics = st.session_state.metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['rmse']:.4f}")
        col2.metric("MAE", f"{metrics['mae']:.4f}")
        col3.metric("R² Score", f"{metrics['r2']:.4f}")
        
        # Plot actual vs predicted
        st.subheader("Actual vs Predicted Overruns")
        fig_pred = px.scatter(
            x=st.session_state.predictions['test']['actual'],
            y=st.session_state.predictions['test']['predicted'],
            labels={"x": "Actual Overrun (%)", "y": "Predicted Overrun (%)"},
            title="Actual vs Predicted Cost Overruns"
        )
        fig_pred.add_trace(
            go.Scatter(
                x=[min(st.session_state.predictions['test']['actual']), 
                   max(st.session_state.predictions['test']['actual'])],
                y=[min(st.session_state.predictions['test']['actual']), 
                   max(st.session_state.predictions['test']['actual'])],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction"
            )
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance
        st.header("Risk Factor Analysis")
        
        st.subheader("Key Risk Factors Contributing to Cost Overruns")
        fig_importance = plot_feature_importance(st.session_state.feature_importance)
        st.pyplot(fig_importance)
        
        # Generate insights
        st.header("Insights and Recommendations")
        insights = generate_insights(st.session_state.feature_importance, st.session_state.metrics)
        st.write(insights)
        
        # Contingency calculator
        st.header("Contingency Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Project Details")
            project_budget = st.number_input("Project Budget (₦ Million)", min_value=1.0, value=100.0)
            
            # Select top 5 features for user input
            top_features = [feature for feature, _ in st.session_state.feature_importance[:5]]
            
            # User inputs for key features
            feature_inputs = {}
            for feature in top_features:
                feature_inputs[feature] = st.slider(
                    f"{feature} (Scale 1-10)", 
                    min_value=1, 
                    max_value=10, 
                    value=5
                )
                
        with col2:
            st.subheader("Contingency Recommendation")
            if st.button("Calculate Contingency"):
                # Convert feature inputs to the right format for prediction
                input_data = st.session_state.processor.prepare_prediction_data(feature_inputs)
                
                # Predict overrun percentage
                predicted_overrun = st.session_state.model.predict(input_data)[0]
                
                # Calculate contingency
                contingency_percent, contingency_amount = calculate_contingency(
                    predicted_overrun, 
                    project_budget,
                    model_accuracy=st.session_state.metrics['r2']
                )
                
                # Display results
                st.metric("Predicted Overrun (%)", f"{predicted_overrun:.2f}%")
                st.metric("Recommended Contingency (%)", f"{contingency_percent:.2f}%")
                st.metric("Contingency Amount (₦ Million)", f"{contingency_amount:.2f}")
                
                # Risk category
                if predicted_overrun <= 10:
                    risk_category = "Low Risk"
                    risk_color = "green"
                elif predicted_overrun <= 25:
                    risk_category = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_category = "High Risk"
                    risk_color = "red"
                    
                st.markdown(f"<h3 style='color:{risk_color}'>{risk_category}</h3>", unsafe_allow_html=True)
                
                # Recommendations based on risk level
                st.subheader("Recommended Actions:")
                if risk_category == "Low Risk":
                    st.write("- Standard monitoring procedures should be sufficient")
                    st.write("- Regular progress reviews at standard intervals")
                    st.write("- Standard documentation and reporting")
                elif risk_category == "Medium Risk":
                    st.write("- Increased monitoring frequency")
                    st.write("- Detailed cost tracking and variance analysis")
                    st.write("- Regular risk reassessment")
                    st.write("- Consider value engineering options")
                else:
                    st.write("- Immediate detailed risk mitigation plan required")
                    st.write("- Weekly monitoring and reporting")
                    st.write("- Consider project rescoping or phasing")
                    st.write("- Expert review of all major cost components")
                    st.write("- Establish formal change management process")

        # Option to download a report
        if st.session_state.model is not None:
            st.header("Download Report")
            
            if st.button("Generate Report"):
                # Create a StringIO object to hold the report content
                report_buffer = io.StringIO()
                
                # Write report content
                report_buffer.write("# Nigerian Construction Project Cost Overrun Report\n\n")
                report_buffer.write(f"## Model Performance\n")
                report_buffer.write(f"- RMSE: {metrics['rmse']:.4f}\n")
                report_buffer.write(f"- MAE: {metrics['mae']:.4f}\n")
                report_buffer.write(f"- R² Score: {metrics['r2']:.4f}\n\n")
                
                report_buffer.write("## Key Risk Factors (Importance)\n")
                for feature, importance in st.session_state.feature_importance:
                    report_buffer.write(f"- {feature}: {importance:.4f}\n")
                
                report_buffer.write("\n## Insights and Recommendations\n")
                report_buffer.write(insights)
                
                # Convert to bytes for download
                report_content = report_buffer.getvalue()
                
                # Create download button
                st.download_button(
                    label="Download Report as Text",
                    data=report_content,
                    file_name="construction_cost_overrun_report.txt",
                    mime="text/plain"
                )
else:
    st.info("Please upload data or select sample data to begin.")
