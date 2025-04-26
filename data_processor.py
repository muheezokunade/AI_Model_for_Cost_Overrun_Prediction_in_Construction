import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    """Class for preprocessing construction project data"""
    
    def __init__(self, data):
        """
        Initialize with dataset
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw construction project data
        """
        self.data = data
        self.preprocessor = None
        self.target_column = self._identify_target_column()
        
    def _identify_target_column(self):
        """
        Identify the target column (cost overrun percentage)
        
        Returns:
        --------
        str
            Name of the target column
        """
        # Look for columns that might contain cost overrun information
        potential_targets = [
            'cost_overrun_percentage', 'cost_overrun', 'overrun_percentage', 
            'overrun', 'cost_variance', 'variance_percentage'
        ]
        
        for col in potential_targets:
            if col in self.data.columns:
                return col
        
        # If no matching column found, use the last column as default
        # This is a fallback and might not be accurate
        return self.data.columns[-1]
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess data for model training
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        X_train, X_test, y_train, y_test, feature_names
            Processed data splits and feature names
        """
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Store original feature names
        feature_names = list(X.columns)
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Apply preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after preprocessing (for interpretation)
        feature_names_processed = self._get_feature_names_after_preprocessing(
            numeric_features, categorical_features
        )
        
        return X_train_processed, X_test_processed, y_train, y_test, feature_names_processed
    
    def _get_feature_names_after_preprocessing(self, numeric_features, categorical_features):
        """
        Get feature names after preprocessing
        
        Parameters:
        -----------
        numeric_features : list
            Names of numeric features
        categorical_features : list
            Names of categorical features
            
        Returns:
        --------
        list
            Feature names after preprocessing
        """
        # For numeric features, names remain the same
        feature_names = list(numeric_features)
        
        # For categorical features, get the one-hot encoded feature names
        if len(categorical_features) > 0:
            try:
                # Get the onehotencoder from the column transformer
                ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                
                # Get the categorical feature names after one-hot encoding
                categories = ohe.categories_
                
                for i, category in enumerate(categories):
                    category_names = [f"{categorical_features[i]}_{cat}" for cat in category]
                    feature_names.extend(category_names)
            except:
                # If something goes wrong, use generic feature names
                for i, feature in enumerate(categorical_features):
                    feature_names.append(f"{feature}_encoded")
        
        return feature_names
    
    def prepare_prediction_data(self, feature_inputs):
        """
        Prepare input data for prediction
        
        Parameters:
        -----------
        feature_inputs : dict
            Dictionary of feature names and values
            
        Returns:
        --------
        array-like
            Processed input data ready for prediction
        """
        # Create a DataFrame with the input features
        input_df = pd.DataFrame([feature_inputs])
        
        # Process the input data using the same preprocessing pipeline
        processed_input = self.preprocessor.transform(input_df)
        
        return processed_input
