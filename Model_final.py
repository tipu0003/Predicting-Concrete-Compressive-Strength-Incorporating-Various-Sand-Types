# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, learning_curve, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

# Feature scaling
from sklearn.preprocessing import StandardScaler

# SHAP for interpretability
import shap

# GUI libraries
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from matplotlib import rc

# Set the font to 'serif' and use 'Computer Modern'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# Enable LaTeX text rendering
rc('text', usetex=True)

# Update other plotting parameters
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 900,
    "savefig.dpi": 900
})

# Function to load and preprocess data
def load_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Assume data is already cleaned and encoded as per user's instructions
    # Separate features and target variables
    X = df.drop(['CS', 'TS'], axis=1)  # Features
    y_cs = df['CS']  # Compressive Strength
    y_ts = df['TS']  # Tensile Strength
    
    return X, y_cs, y_ts

# Function to scale features
def scale_features(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled

# Function to perform hyperparameter tuning and model training
def train_models(X, y):
    models = {}
    model_performance = {}
    hyperparameters = {}
    default_hyperparameters = {}
    
    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    default_hyperparameters['Linear Regression'] = lr.get_params()
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='neg_mean_squared_error')
    lr_mse = -lr_scores.mean()
    models['Linear Regression'] = lr
    model_performance['Linear Regression'] = lr_mse
    hyperparameters['Linear Regression'] = lr.get_params()
    
    # Support Vector Regression
    svr = SVR()
    default_hyperparameters['SVR'] = svr.get_params()
    svr_params = {
        'kernel': ['rbf'],
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    svr_grid = GridSearchCV(svr, svr_params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    svr_grid.fit(X, y)
    svr_best = svr_grid.best_estimator_
    svr_mse = -svr_grid.best_score_
    models['SVR'] = svr_best
    model_performance['SVR'] = svr_mse
    hyperparameters['SVR'] = svr_grid.best_params_
    
    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    default_hyperparameters['Random Forest'] = rf.get_params()
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X, y)
    rf_best = rf_grid.best_estimator_
    rf_mse = -rf_grid.best_score_
    models['Random Forest'] = rf_best
    model_performance['Random Forest'] = rf_mse
    hyperparameters['Random Forest'] = rf_grid.best_params_
    
    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42)
    default_hyperparameters['GBR'] = gbr.get_params()
    gbr_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    gbr_grid = GridSearchCV(gbr, gbr_params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    gbr_grid.fit(X, y)
    gbr_best = gbr_grid.best_estimator_
    gbr_mse = -gbr_grid.best_score_
    models['GBR'] = gbr_best
    model_performance['GBR'] = gbr_mse
    hyperparameters['GBR'] = gbr_grid.best_params_
    
    # XGBoost
    xgb = XGBRegressor(random_state=42)
    default_hyperparameters['XGBoost'] = xgb.get_params()
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_grid.fit(X, y)
    xgb_best = xgb_grid.best_estimator_
    xgb_mse = -xgb_grid.best_score_
    models['XGBoost'] = xgb_best
    model_performance['XGBoost'] = xgb_mse
    hyperparameters['XGBoost'] = xgb_grid.best_params_
    
    # MLP Regressor
    mlp = MLPRegressor(random_state=42, max_iter=1000)
    default_hyperparameters['MLP'] = mlp.get_params()
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    }
    mlp_grid = GridSearchCV(mlp, mlp_params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    mlp_grid.fit(X, y)
    mlp_best = mlp_grid.best_estimator_
    mlp_mse = -mlp_grid.best_score_
    models['MLP'] = mlp_best
    model_performance['MLP'] = mlp_mse
    hyperparameters['MLP'] = mlp_grid.best_params_
    
    return models, model_performance, hyperparameters, default_hyperparameters

# Function to evaluate models
def evaluate_models(models, X, y):
    performance_metrics = {}
    predictions = {}
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=5)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        performance_metrics[name] = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        predictions[name] = y_pred
    return performance_metrics, predictions

# Function to plot learning curves and record training samples
def plot_learning_curves(models, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    learning_curve_data = {}
    for name, model in models.items():
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        # Record learning curve data
        learning_curve_data[name] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores_mean,
            'test_scores': test_scores_mean
        }
        
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation error")
        plt.title(f'Learning Curve for {name}')
        plt.xlabel('Training examples')
        plt.ylabel('Error')
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(f'learning_curve_{name}.png', dpi=900, bbox_inches='tight')
        plt.close()
    return learning_curve_data

# Function to compute ensemble predictions and weights
def compute_ensemble_weights(models, X, y):
    # Compute inverse of RMSE as weights
    weights = {}
    total_inv_rmse = 0
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        if model is None:
            continue  # Skip Ensemble placeholder
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        inv_rmse = 1 / rmse
        weights[name] = inv_rmse
        total_inv_rmse += inv_rmse
    
    # Normalize weights
    for name in weights:
        weights[name] /= total_inv_rmse
    
    return weights

# Function to compute ensemble predictions
def ensemble_predictions(models, weights, X, y):
    # Get predictions from each model
    preds = {}
    for name, model in models.items():
        if model is None:
            continue  # Skip Ensemble placeholder
        model.fit(X, y)
        preds[name] = model.predict(X)
    
    # Compute weighted average
    ensemble_pred = np.zeros_like(y)
    for name in preds:
        ensemble_pred += weights[name] * preds[name]
    
    return ensemble_pred, preds

# Function to perform SHAP analysis with actual feature names
def shap_analysis(model, X_scaled, feature_names):
    # Convert X_scaled back to DataFrame with feature names
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_df)
    
    # Compute SHAP values
    shap_values = explainer(X_df)
    
    # Summary plot with actual feature names
    shap.summary_plot(shap_values, X_df, show=False)
    plt.savefig('shap_summary_plot.png', dpi=900, bbox_inches='tight')
    plt.close()
    
    # Optional: Save SHAP values to a file for further analysis
    # shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
    # shap_values_df.to_csv('shap_values.csv', index=False)

# Function to export results to Excel
def export_results_to_excel(performance_metrics, weights, hyperparameters, default_hyperparameters, learning_curve_data, predictions, y_actual):
    with pd.ExcelWriter('model_performance.xlsx') as writer:
        # Performance metrics
        perf_df = pd.DataFrame(performance_metrics).transpose()
        perf_df.to_excel(writer, sheet_name='Performance')
        
        # Model Weights
        weights_df = pd.DataFrame(list(weights.items()), columns=['Model', 'Weight'])
        weights_df.to_excel(writer, sheet_name='Model Weights', index=False)
        
        # Hyperparameters
        hyper_df = pd.DataFrame(hyperparameters).transpose()
        hyper_df.to_excel(writer, sheet_name='Optimized Hyp')
        
        # Default Hyperparameters
        default_hyper_df = pd.DataFrame(default_hyperparameters).transpose()
        default_hyper_df.to_excel(writer, sheet_name='Default Hyp')
        
        # Learning Curve Data
        for name, data in learning_curve_data.items():
            lc_df = pd.DataFrame({
                'Training Samples': data['train_sizes'],
                'Training Error': data['train_scores'],
                'Cross-validation Error': data['test_scores']
            })
            lc_df.to_excel(writer, sheet_name=f'LC {name}')
        
        # Actual and Predicted Values
        for name, y_pred in predictions.items():
            actual_vs_pred = pd.DataFrame({
                'Actual': y_actual,
                'Predicted': y_pred
            })
            actual_vs_pred.to_excel(writer, sheet_name=f'Predictions {name}', index=False)

# GUI implementation for user input and prediction with Scrollbar
def run_prediction_gui(models, scaler, feature_names, weights, X_scaled, y_cs):
    root = tk.Tk()
    root.title("Concrete Strength Prediction")
    root.geometry("500x600")  # Set an initial size; adjust as needed

    # Create a frame for the canvas and scrollbar
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create a canvas
    canvas = tk.Canvas(canvas_frame, borderwidth=0, background="#f0f0f0")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a vertical scrollbar to the canvas
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure the canvas to work with the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Create a frame inside the canvas to hold the input widgets
    scrollable_frame = ttk.Frame(canvas, padding=(10, 10, 10, 10))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Add input fields to the scrollable_frame
    entries = {}
    for i, feature in enumerate(feature_names):
        label = ttk.Label(scrollable_frame, text=feature)
        label.grid(row=i, column=0, sticky=tk.W, pady=5, padx=5)
        entry = ttk.Entry(scrollable_frame)
        entry.grid(row=i, column=1, pady=5, padx=5)
        entries[feature] = entry

    # Function to predict based on user input
    def predict():
        try:
            # Collect inputs
            input_data = []
            for feature in feature_names:
                value = float(entries[feature].get())
                input_data.append(value)
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            # Get predictions from each model
            predictions = {}
            for name, model in models.items():
                if name == 'Ensemble' or model is None:
                    continue
                pred = model.predict(input_scaled)[0]
                predictions[name] = pred
            
            # Ensemble prediction using precomputed weights
            ensemble_pred = 0
            for name in predictions:
                ensemble_pred += weights[name] * predictions[name]
            predictions['Ensemble'] = ensemble_pred
            
            # Display predictions
            result_text = "\nPredicted Compressive Strength (CS):\n"
            for name, pred in predictions.items():
                result_text += f"{name}: {pred:.2f}\n"
            messagebox.showinfo("Prediction Results", result_text)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numerical values for all features.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Predict button
    predict_button = ttk.Button(root, text="Predict CS", command=predict)
    predict_button.pack(pady=10)

    # Make sure the scrollable_frame resizes properly
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    root.mainloop()

# Main function to run the entire pipeline
def main():
    # Load data
    X, y_cs, y_ts = load_data('cleaned_encoded_data.csv')
    
    # Scale features
    scaler, X_scaled = scale_features(X)
    
    # Train models for Compressive Strength
    print("Training models for Compressive Strength...")
    models_cs, model_perf_cs, hyperparameters_cs, default_hyperparameters_cs = train_models(X_scaled, y_cs)
    
    # Evaluate models
    print("Evaluating models for Compressive Strength...")
    perf_metrics_cs, predictions_cs = evaluate_models(models_cs, X_scaled, y_cs)
    
    # Plot learning curves
    print("Plotting learning curves for Compressive Strength...")
    learning_curve_data_cs = plot_learning_curves(models_cs, X_scaled, y_cs)
    
    # Compute ensemble weights
    print("Computing ensemble weights for Compressive Strength...")
    weights_cs = compute_ensemble_weights(models_cs, X_scaled, y_cs)
    
    # Compute ensemble predictions
    print("Computing ensemble predictions for Compressive Strength...")
    ensemble_pred_cs, preds_cs = ensemble_predictions(models_cs, weights_cs, X_scaled, y_cs)
    
    # Evaluate ensemble model
    mse_ensemble_cs = mean_squared_error(y_cs, ensemble_pred_cs)
    mae_ensemble_cs = mean_absolute_error(y_cs, ensemble_pred_cs)
    mape_ensemble_cs = mean_absolute_percentage_error(y_cs, ensemble_pred_cs)
    r2_ensemble_cs = r2_score(y_cs, ensemble_pred_cs)
    perf_metrics_cs['Ensemble'] = {
        'MSE': mse_ensemble_cs,
        'MAE': mae_ensemble_cs,
        'MAPE': mape_ensemble_cs,
        'R2': r2_ensemble_cs
    }
    predictions_cs['Ensemble'] = ensemble_pred_cs
    
    # Update models and hyperparameters with Ensemble
    models_cs['Ensemble'] = None  # Placeholder
    hyperparameters_cs['Ensemble'] = {'Weights': weights_cs}
    default_hyperparameters_cs['Ensemble'] = {'Weights': weights_cs}
    
    # SHAP analysis on a representative model (e.g., XGBoost) with actual feature names
    print("Performing SHAP analysis for Compressive Strength...")
    shap_analysis(models_cs['XGBoost'], X_scaled, X.columns)
    
    # Export results to Excel
    print("Exporting results to Excel...")
    export_results_to_excel(
        performance_metrics=perf_metrics_cs,
        weights=weights_cs,
        hyperparameters=hyperparameters_cs,
        default_hyperparameters=default_hyperparameters_cs,
        learning_curve_data=learning_curve_data_cs,
        predictions=predictions_cs,
        y_actual=y_cs
    )
    
    # GUI for user input to predict CS
    print("Launching GUI for predictions...")
    run_prediction_gui(models_cs, scaler, X.columns, weights_cs, X_scaled, y_cs)
    
    print("All tasks completed successfully.")

if __name__ == "__main__":
    # Run the main analysis
    main()
