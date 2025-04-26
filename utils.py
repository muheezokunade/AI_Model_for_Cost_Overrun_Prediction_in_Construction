import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(feature_importances, top_n=10):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_importances : list of tuples
        List of (feature_name, importance_score) tuples
    top_n : int
        Number of top features to display
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Get top N features
    top_features = feature_importances[:top_n]
    
    # Extract feature names and importance scores
    features = [feature[0] for feature in top_features]
    importances = [feature[1] for feature in top_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Top Features Contributing to Cost Overruns')
    
    plt.tight_layout()
    return fig

def generate_insights(feature_importance, metrics):
    """
    Generate insights based on model results
    
    Parameters:
    -----------
    feature_importance : list of tuples
        List of (feature_name, importance_score) tuples
    metrics : dict
        Dictionary of model performance metrics
        
    Returns:
    --------
    str
        Insights and recommendations text
    """
    # Get top 5 risk factors
    top_factors = [factor[0] for factor in feature_importance[:5]]
    
    # Generate insights text
    insights = f"""
    ## Key Insights
    
    The model has identified the following top risk factors for cost overruns in Nigerian construction projects:
    
    1. **{top_factors[0]}** - This is the most significant factor influencing cost overruns.
    2. **{top_factors[1]}** - Second most important factor.
    3. **{top_factors[2]}** - Third most important factor.
    4. **{top_factors[3]}** - Fourth most important factor.
    5. **{top_factors[4]}** - Fifth most important factor.
    
    The model has an R² score of {metrics['r2']:.4f}, which means it explains approximately {metrics['r2']*100:.1f}% of the variance in cost overruns.
    
    ## Recommendations
    
    Based on these findings, we recommend:
    
    1. Increased focus on monitoring and controlling the top risk factors identified
    2. Implement early warning systems for projects with high-risk profiles
    3. Develop standardized risk assessment procedures that emphasize these key factors
    4. Allocate contingency funds proportionally to the risk level, with a minimum of 10% for low-risk projects
    5. Conduct regular reviews throughout the project lifecycle to reassess risk factors
    """
    
    return insights

def calculate_contingency(predicted_overrun, project_budget, model_accuracy=0.7):
    """
    Calculate recommended contingency allocation
    
    Parameters:
    -----------
    predicted_overrun : float
        Predicted overrun percentage
    project_budget : float
        Total project budget
    model_accuracy : float
        Model accuracy (R² score)
        
    Returns:
    --------
    tuple
        (contingency_percentage, contingency_amount)
    """
    # Base contingency on predicted overrun
    base_contingency = predicted_overrun
    
    # Add safety factor based on model accuracy
    # Lower accuracy = higher safety factor
    safety_factor = 1.0 + (1.0 - model_accuracy)
    
    # Calculate contingency percentage
    contingency_percentage = base_contingency * safety_factor
    
    # Ensure minimum contingency of 5%
    contingency_percentage = max(contingency_percentage, 5.0)
    
    # Calculate contingency amount
    contingency_amount = project_budget * (contingency_percentage / 100.0)
    
    return contingency_percentage, contingency_amount
