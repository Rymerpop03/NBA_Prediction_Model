import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
notebooks_dir = os.path.join(project_root, 'Notebooks')
data_dir = os.path.join(project_root, 'data', 'processed')

models = {}
team_averages = {}
scalers = {}

try:
    # Load Random Forest model
    models['rf'] = joblib.load(os.path.join(notebooks_dir, 'rf_model.pkl'))
    team_averages['rf'] = pd.read_csv(os.path.join(data_dir, 'team_averages_rf.csv'), index_col=0)
    
    # Load XGBoost model
    models['xgb'] = joblib.load(os.path.join(notebooks_dir, 'xgb_model.pkl'))
    team_averages['xgb'] = pd.read_csv(os.path.join(data_dir, 'team_averages_xgb.csv'), index_col=0)
    
    # Load KNN model and its scaler
    models['knn'] = joblib.load(os.path.join(notebooks_dir, 'knn_model.pkl'))
    scalers['knn'] = joblib.load(os.path.join(notebooks_dir, 'knn_scaler.pkl'))
    team_averages['knn'] = pd.read_csv(os.path.join(data_dir, 'team_averages_knn.csv'), index_col=0)
    
    print("Successfully loaded all models: Random Forest, XGBoost, and KNN")
    
except Exception as e:
    print(f"Error loading models or data: {e}")
    print("Please ensure all model files and data are in the correct directories.")
    exit(1)

def predict_matchup(team1_abbr, team2_abbr, model_type, is_team1_home=True):
    """Make prediction using the specified model type ('rf', 'xgb', or 'knn')"""
    
    clf = models[model_type]
    team_avgs = team_averages[model_type]
    
    # Use season averages to form a feature vector (team1 - team2)
    if team1_abbr not in team_avgs.index or team2_abbr not in team_avgs.index:
        raise ValueError(f'Missing averages for {team1_abbr} or {team2_abbr}')
        
    # Use the same stats we trained with
    stats_to_use = ['r50_MissedFG','r50_MissedFT','r50_TSA','r50_TS_Pct','r50_EffIndex','r50_FG_Eff','r50_RebRatio','r50_TS']
    
    # Get team stats
    f1 = team_avgs.loc[team1_abbr, stats_to_use].astype(float)
    f2 = team_avgs.loc[team2_abbr, stats_to_use].astype(float)
    
    # Calculate differences and add home court advantage
    diff_values = (f1 - f2).values
    features = np.append(diff_values, [1 if is_team1_home else 0])
    features = features.reshape(1, -1)
    
    # Apply scaling for KNN model
    if model_type == 'knn':
        features = scalers['knn'].transform(features)
    
    # Make prediction
    prob_team1 = float(clf.predict_proba(features)[0][1])
    pred = int(clf.predict(features)[0])
    winner = team1_abbr if pred == 1 else team2_abbr
    return {'winner': winner, 'probability_team1_win': prob_team1}

def show_available_teams():
    print("\nAvailable teams for prediction:")
    available_teams = sorted(team_averages['rf'].index.tolist())
    for i, team in enumerate(available_teams):
        print(f"{team:<5}", end=" ")
        if (i + 1) % 6 == 0: 
            print()
    print("\n")

def show_model_options():
    print("\nAvailable prediction models:")
    print("1. Random Forest (rf)")
    print("2. XGBoost (xgb)")
    print("3. K-Nearest Neighbors (knn)")
    return {'1': 'rf', '2': 'xgb', '3': 'knn'}

def interactive_prediction():
    show_available_teams()
    
    # Get model selection
    model_choices = show_model_options()
    while True:
        model_input = input("\nSelect model (1-3) or 'q' to quit: ")
        if model_input.lower() == 'q':
            return
        if model_input in model_choices:
            selected_model = model_choices[model_input]
            model_names = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'knn': 'K-Nearest Neighbors'}
            print(f"Selected model: {model_names[selected_model]}")
            break
        else:
            print("Please enter 1, 2, or 3 to select a model.")
    
    # Get team averages for validation (use any dataset as they have same teams)
    available_teams = team_averages['rf'].index.tolist()
    
    while True:
        # Get team inputs
        team1 = input("\nEnter first team abbreviation (or 'q' to quit): ").upper()
        if team1 == 'Q':
            break
            
        team2 = input("Enter second team abbreviation: ").upper()
        
        # Get home team info
        is_team1_home = input(f"Is {team1} the home team? (y/n): ").lower().startswith('y')
            
        # Validate inputs
        if team1 not in available_teams or team2 not in available_teams:
            print(f"Error: Please use valid team abbreviations from the list above.")
            continue
            
        # Make prediction
        try:
            result = predict_matchup(team1, team2, selected_model, is_team1_home)
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"Model: {model_names[selected_model]}")
            print(f"Matchup: {team1} {'(Home)' if is_team1_home else '(Away)'} vs {team2} {'(Home)' if not is_team1_home else '(Away)'}")
            print(f"Predicted Winner: {result['winner']}")
            print(f"Win Probability for {team1}: {result['probability_team1_win']:.1%}")
            print(f"Win Probability for {team2}: {(1 - result['probability_team1_win']):.1%}")
            print("-" * 50)
        except Exception as e:
            print(f"Error making prediction: {e}")
        
        # Ask if user wants to continue with same model or change
        continue_choice = input("\nEnter another matchup (m), change model (c), or quit (q): ").lower()
        if continue_choice == 'q':
            break
        elif continue_choice == 'c':
            interactive_prediction()
            break

if __name__ == "__main__":
    interactive_prediction()