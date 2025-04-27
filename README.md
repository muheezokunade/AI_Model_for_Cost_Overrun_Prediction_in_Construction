# Nigerian Construction Project Cost Overrun Predictor

## Overview
This machine learning application predicts cost overruns in Nigerian construction projects and recommends appropriate contingency allocations based on project characteristics. By analyzing historical data from public infrastructure projects in Nigeria, the model identifies key risk factors that contribute to cost overruns and helps project planners make data-driven decisions.

## Features
- **Data Analysis**: Upload your own CSV data or use the included sample dataset of Nigerian construction projects
- **Multiple ML Models**: Choose from Linear Regression, Random Forest, Gradient Boosting, or Support Vector Regression algorithms
- **Model Evaluation**: View performance metrics including RMSE, MAE, and R² score
- **Risk Factor Analysis**: Identify the most significant factors contributing to cost overruns
- **Contingency Calculator**: Input project details to receive recommendations on appropriate contingency allocations
- **Report Generation**: Download analysis reports with key insights and recommendations

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn/Plotly**: Data visualization

## Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/nigerian-construction-cost-predictor.git
cd nigerian-construction-cost-predictor
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

## Usage
1. **Data Input**: 
   - Select "Use Sample Data" for demonstration or "Upload CSV" to use your own dataset
   - The CSV should contain project information and a target column indicating cost overrun percentage

2. **Model Training**:
   - Select a model type from the dropdown menu
   - Adjust the test size percentage as needed
   - Click "Train Model" to process the data and train the selected algorithm

3. **Results Analysis**:
   - View model performance metrics and the actual vs. predicted visualization
   - Examine key risk factors contributing to cost overruns
   - Read generated insights and recommendations

4. **Contingency Planning**:
   - Enter your project budget and adjust risk factor sliders
   - Click "Calculate Contingency" to receive a recommended contingency allocation
   - View risk category and recommended actions based on the prediction

5. **Report Generation**:
   - Click "Generate Report" to create a downloadable text report of findings

## Project Structure
- `app.py`: Main Streamlit application
- `data_processor.py`: Data preprocessing functionality
- `model.py`: Machine learning model implementation
- `utils.py`: Utility functions for visualization and insights
- `sample_data.csv`: Example dataset for demonstration
- `.streamlit/config.toml`: Streamlit configuration settings

## Data Requirements
For custom data uploads, CSV files should include:
- Project characteristics (size, duration, location, etc.)
- Initial budget and final cost figures
- Risk factors such as weather delays, design changes, etc.
- Cost overrun percentage (target variable)

## Future Improvements
- Integration of more advanced deep learning models
- Regional analysis for different Nigerian states
- Time-series analysis of cost trends
- Integration with project management tools
- Expanded visualization options

## License
[Add appropriate license information]

## Contributors
[Your name/organization]

## Acknowledgments
- Nigerian construction industry data providers
- [Add any other acknowledgments]