import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LassoCV,LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
def forward_selection(X, y, constant=False):
    """
    Performs forward selection on the given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Independent variables of the dataset.
    y : pandas.Series or numpy.array
        Dependent variable of the dataset.
    constant : bool, optional
        If True, adds a constant term to the model. The default is False.

    Returns
    -------
    sel_features : list
        List of indices corresponding to the selected features.

    Notes
    -----
    This function performs forward selection using Ordinary Least Squares (OLS) regression. 
    It starts with no variables in the model, and in each iteration, it adds the variable 
    that has the lowest p-value, provided it is less than the given significance level (alpha). 
    The function stops when no variables can be added that meet the alpha criterion.

    """
    # Initialize selected features and candidates
    sel_features = []
    candidates = list(range(X.shape[1]))

    # Set the significance level
    alpha = 0.05

    while len(candidates) > 0:
        pvalues = []
        
        for candidate in candidates:
            # Create a temporary feature set including the current candidate
            temp_features = sel_features + [candidate]
            
            if constant:
                # Add a constant to the dataset for the intercept
                X_temp = sm.add_constant(X.iloc[:, temp_features])
            else:
                X_temp = X.iloc[:, temp_features]

            # Fit the OLS model
            model = sm.OLS(y, X_temp).fit()

            # Get the p-value of the candidate
            pvalues.append(model.pvalues[-1])

        # Find the index of the lowest p-value
        best_feature_idx = np.argmin(pvalues)

        # If the lowest p-value is less than the significance level, add the feature to the selected features
        if pvalues[best_feature_idx] < alpha:
            selected_feature = candidates.pop(best_feature_idx)
            sel_features.append(selected_feature)
        else:
            # If no more features have p-values less than alpha, break the loop
            break

    return sel_features

def backward_elimination(X, y, constant=False):
    """
    Performs backward elimination on the given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Independent variables of the dataset.
    y : pandas.Series or numpy.array
        Dependent variable of the dataset.
    constant : bool, optional
        If True, adds a constant term to the model. The default is False.

    Returns
    -------
    sel_features : list
        List of indices corresponding to the selected features.

    Notes
    -----
    This function performs backward elimination using Ordinary Least Squares (OLS) regression. 
    It starts with all variables in the model, and in each iteration, it removes the variable 
    that has the highest p-value, provided it is greater than the given significance level (alpha). 
    The function stops when no variables can be removed that meet the alpha criterion.

    """
    # Initialize selected features
    sel_features = list(range(X.shape[1]))

    # Set the significance level
    alpha = 0.05

    while True:
        if constant:
            # Add a constant to the dataset for the intercept
            X_temp = sm.add_constant(X.iloc[:, sel_features])
        else:
            X_temp = X.iloc[:, sel_features]

        # Fit the OLS model
        model = sm.OLS(y, X_temp).fit()

        # Get the p-values of the features
        pvalues = model.pvalues[1:] if constant else model.pvalues

            # Find the index of the highest p-value
        worst_feature_idx = np.argmax(pvalues)

            # If the highest p-value is greater than the significance level, remove the feature from the selected features
        if pvalues[worst_feature_idx] > alpha:
                removed_feature = sel_features.pop(worst_feature_idx)
        else:
                # If no more features have p-values greater than alpha, break the loop
                break

    return sel_features

def bidirect_elimination(X, y, constant=False):
    """
    Performs bidirectional elimination (also known as stepwise selection) on the given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Independent variables of the dataset.
    y : pandas.Series or numpy.array
        Dependent variable of the dataset.
    constant : bool, optional
        If True, adds a constant term to the model. The default is False.

    Returns
    -------
    sel_features : list
        List of indices corresponding to the selected features.

    Notes
    -----
    This function performs bidirectional elimination using Ordinary Least Squares (OLS) regression. 
    It follows a combination of forward selection and backward elimination. In each iteration, 
    it adds the variable that has the lowest p-value (if it's less than alpha_in) and removes 
    the variable that has the highest p-value (if it's greater than alpha_out). 
    The function stops when no variables can be added or removed.

    """
    # Initialize selected features and candidates
    sel_features = []
    candidates = list(range(X.shape[1]))

    # Set the significance level
    alpha_in = 0.05 
    alpha_out = 0.1

    while True:
        # Forward Selection
        forward_candidates = candidates.copy()
        forward_changed = False
        for candidate in forward_candidates:
            temp_features = sel_features + [candidate]
            if constant:
                X_temp = sm.add_constant(X.iloc[:, temp_features])
            else:
                X_temp = X.iloc[:, temp_features]

            model = sm.OLS(y, X_temp).fit()

            if model.pvalues[-1] < alpha_in:
                sel_features.append(candidate)
                candidates.remove(candidate)
                forward_changed = True

        # Backward Elimination
        backward_changed = False
        backward_candidates = sel_features.copy()
        for candidate in backward_candidates:
            if constant:
                X_temp = sm.add_constant(X.iloc[:, sel_features])
            else:
                X_temp = X.iloc[:, sel_features]

            model = sm.OLS(y, X_temp).fit()

            candidate_idx = sel_features.index(candidate)
            if model.pvalues[candidate_idx + int(constant)] > alpha_out:
                sel_features.remove(candidate)
                candidates.append(candidate)
                backward_changed = True

        # If no features were added or removed during this iteration, stop the loop
        if not forward_changed and not backward_changed:
            break
    return sel_features

def fit_ols_model(selected_features, model_name, df, dependent, constant=False):
    """
    Fits an Ordinary Least Squares (OLS) model on the given dataset with the selected features.

    Parameters
    ----------
    selected_features : list
        List of indices corresponding to the selected features.
    model_name : str
        Name of the model for displaying purposes.
    df : pandas.DataFrame
        DataFrame containing the dependent and independent variables.
    dependent : str
        Name of the dependent variable.
    constant : bool, optional
        If True, adds a constant term to the model. The default is False.

    Returns
    -------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.

    Prints
    ------
    Adjusted R^2 value of the fitted model.
    
    Notes
    -----
    This function fits an OLS regression model on the given dataset with the selected features 
    and prints the adjusted R^2 value of the model. The fitted model is returned.

    """
    X=df[df.columns[selected_features]]
    y=df[dependent]
    if constant:
        X=sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    adjusted_r2 = model.rsquared_adj
    print(f"Adjusted R^2 ({model_name}): {adjusted_r2}")
    return model

def run_analysis(df, dependent, split="", constant=False):
    """
    Runs an analysis by performing feature selection, fitting OLS models, and printing model summaries.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the dependent and independent variables.
    split : str, optional
        String to be appended to the model names for display purposes. The default is "".
    dependent : str, optional
        Name of the dependent variable. The default is 'LGDobs'.
    constant : bool, optional
        If True, adds a constant term to the model. The default is False.

    Returns
    -------
    forward_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model using forward selection.
    bidirect_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model using bidirectional elimination.
    backward_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model using backward elimination.

    Prints
    ------
    Selected features for each method and summaries of the fitted models.

    Notes
    -----
    This function performs feature selection using forward selection, bidirectional elimination, 
    and backward elimination. It then fits an OLS regression model for each set of selected features, 
    prints the selected features and model summaries, and returns the fitted models.

    """
    # rest of your code follows here

    df = df[[col for col in df if col != dependent]+[dependent]]
    # Split the dataset into X and y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Perform feature selection
    forward_sel = forward_selection(X, y, constant=constant)
    bidirect_sel = bidirect_elimination(X, y, constant=constant)
    backward_sel = backward_elimination(X, y, constant=constant)

    # Print the selected features
    print(f"Selected features (forward selection) {split}:", forward_sel)
    print(f"Selected features (bidirectional elimination) {split}:", bidirect_sel)
    print(f"Selected features (backward elimination) {split}:", backward_sel)

    # Fit the models and print the summaries
    forward_model = fit_ols_model(selected_features=forward_sel, model_name=f"Forward Selection {split}",
                                  df=df,dependent=dependent, constant=constant)
    bidirect_model = fit_ols_model(selected_features=bidirect_sel,model_name= f"Bidirectional Elimination {split}",
                                   df=df, dependent=dependent, constant=constant)
    backward_model = fit_ols_model(selected_features=backward_sel,model_name= f"Backward Elimination {split}",
                                   df=df, dependent=dependent, constant=constant)
    print(forward_model.summary())
    print("")
    print(bidirect_model.summary())
    print("")
    print(backward_model.summary())
    print("")
    return forward_model, bidirect_model,backward_model

#Cross-validation
def cross_val_no_intercept(X, y, splits=5, method=""):
    """
    Performs cross-validation on the data without using an intercept in the OLS model and calculates the mean adjusted R^2.

    Parameters
    ----------
    X : pandas.DataFrame
        Independent variables for the OLS regression.
    y : pandas.Series
        Dependent variable for the OLS regression.
    splits : int, optional
        Number of folds to use for cross-validation. Default is 5.
    method : str, optional
        Method name to print along with the mean adjusted R^2 score. Default is an empty string.

    Returns
    -------
    None

    Prints
    ------
    Mean adjusted R^2 score along with the method name.
    Adjusted R^2 score for each fold.

    Notes
    -----
    This function uses the sklearn's KFold for cross-validation and statsmodels' OLS for regression.
    An inner function adjusted_r2 is defined to calculate the adjusted R^2.
    """
    # Define a function to calculate adjusted R^2
    def adjusted_r2(y_true, y_pred, p):
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adj_r2


    # Set up 5-fold cross-validation
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    # List to store each fold's adjusted R^2
    adjusted_r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = sm.OLS(y_train, X_train)
        results = model.fit()

        y_pred = results.predict(X_test)
        adj_r2 = adjusted_r2(y_test, y_pred, p=X_train.shape[1])
        adjusted_r2_scores.append(adj_r2)

    # Compute the mean adjusted R^2
    mean_adjusted_r2 = np.mean(adjusted_r2_scores)
    print(f"Mean adjusted R^2 score:{method}", mean_adjusted_r2)
    print(adjusted_r2_scores)
