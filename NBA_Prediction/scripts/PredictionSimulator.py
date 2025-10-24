import pandas as pd
import joblib
import numpy as np

# Load the trained model and team averages
try:
    clf = joblib.load('rf_model.pkl')
    team_avgs = pd.read_csv('C:/Users/poke5/Desktop/Projects/NBA_Prediction/data/processed/team_averages.csv', index_col=0)
    features = team_avgs.columns.tolist()
except Exception as e:
    print(f"Error loading model or data: {e}")
    exit(1)

def predict_matchup(team1_abbr, team2_abbr, is_team1_home=True):
    # Use season averages to form a feature vector (team1 - team2)
    if team1_abbr not in team_avgs.index or team2_abbr not in team_avgs.index:
        raise ValueError(f'Missing averages for {team1_abbr} or {team2_abbr}')
        
    # Use the same stats we trained with
    stats_to_use = ['PTS', 'FG_PCT', 'FG3M', 'FT_PCT', 'REB', 'AST', 'TOV']
    
    # Get team stats
    f1 = team_avgs.loc[team1_abbr, stats_to_use].astype(float)
    f2 = team_avgs.loc[team2_abbr, stats_to_use].astype(float)
    
    # Calculate differences and add home court advantage
    diff_values = (f1 - f2).values
    features = np.append(diff_values, [1 if is_team1_home else 0])
    features = features.reshape(1, -1)
    
    # Make prediction
    prob_team1 = float(clf.predict_proba(features)[0][1])
    pred = int(clf.predict(features)[0])
    winner = team1_abbr if pred == 1 else team2_abbr
    return {'winner': winner, 'probability_team1_win': prob_team1}

def show_available_teams():
    print("\nAvailable teams for prediction:")
    available_teams = sorted(team_avgs.index.tolist())
    for i, team in enumerate(available_teams):
        print(f"{team:<5}", end=" ")
        if (i + 1) % 6 == 0: 
            print()
    print("\n")

def interactive_prediction():
    show_available_teams()
    
    while True:
        # Get team inputs
        team1 = input("\nEnter first team abbreviation (or 'q' to quit): ").upper()
        if team1 == 'Q':
            break
            
        team2 = input("Enter second team abbreviation: ").upper()
        
        # Get home team info
        is_team1_home = input(f"Is {team1} the home team? (y/n): ").lower().startswith('y')
            
        # Validate inputs
        if team1 not in team_avgs.index or team2 not in team_avgs.index:
            print(f"Error: Please use valid team abbreviations from the list above.")
            continue
            
        # Make prediction
        try:
            result = predict_matchup(team1, team2, is_team1_home)
            print("\nPrediction Results:")
            print("-" * 40)
            print(f"Matchup: {team1} {'(Home)' if is_team1_home else '(Away)'} vs {team2} {'(Home)' if not is_team1_home else '(Away)'}")
            print(f"Predicted Winner: {result['winner']}")
            print(f"Win Probability for {team1}: {result['probability_team1_win']:.1%}")
            print(f"Win Probability for {team2}: {(1 - result['probability_team1_win']):.1%}")
            print("-" * 40)
        except Exception as e:
            print(f"Error making prediction: {e}")
        
        print("\nEnter another matchup or 'q' to quit.")

if __name__ == "__main__":
    interactive_prediction()
