import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelTrainer:
    """Class for training and evaluating machine learning models for cost overrun prediction"""
    
    def __init__(self, model_type="Random Forest"):
        """
        Initialize model based on specified type
        
        Parameters:
        -----------
        model_type : str
            Type of model to train (Linear Regression, Random Forest, 
            Gradient Boosting, Support Vector Regression, Neural Network)
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the machine learning model based on the selected type
        
        Returns:
        --------
        model : object
            Initialized model object
        """
        if self.model_type == "Linear Regression":
            return LinearRegression()
        elif self.model_type == "Random Forest":
            return RandomForestRegressor(
                n_estimators=100, 
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == "Support Vector Regression":
            return SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1)
        elif self.model_type == "Neural Network":
            # Temporarily using Random Forest as a fallback for Neural Network
            # to avoid TensorFlow dependency issues
            return RandomForestRegressor(
                n_estimators=150, 
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        else:
            # Default to Random Forest if type is not recognized
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the model on the input data
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Target values for training
        epochs : int
            Number of epochs (only used for Neural Network)
        batch_size : int
            Batch size (only used for Neural Network)
        """
        # All models including Neural Network (which is now Random Forest) use the same training method
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : array-like
            Input features for prediction
            
        Returns:
        --------
        array-like
            Predicted values
        """
        # All models use the same prediction method now
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance using various metrics
        
        Parameters:
        -----------
        y_true : array-like
            Actual target values
        y_pred : array-like
            Predicted target values
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Mean Absolute Percentage Error
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importances from the trained model
        
        Parameters:
        -----------
        feature_names : list
            Names of the features
            
        Returns:
        --------
        list of tuples
            (feature_name, importance_score) sorted by importance
        """
        if self.model_type in ["Linear Regression"]:
            importances = np.abs(self.model.coef_)
        elif self.model_type in ["Random Forest", "Gradient Boosting"]:
            importances = self.model.feature_importances_
        elif self.model_type in ["Support Vector Regression", "Neural Network"]:
            # These models don't provide feature importance directly
            # Using permutation importance would be better but requires additional data
            # For now, returning None
            return [(name, 0) for name in feature_names]
        else:
            return [(name, 0) for name in feature_names]
        
        # Create a list of (feature, importance) tuples
        feature_importances = [(name, imp) for name, imp in zip(feature_names, importances)]
        
        # Sort by importance (descending)
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importances
