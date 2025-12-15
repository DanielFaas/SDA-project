import pytest
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# FIXTURES (Create Mock Data)
@pytest.fixture
def mock_chess_data():
    """
    Creates a synthetic dataframe representing your chess dataset.
    Includes edge cases and missing values to test robustness.
    """
    np.random.seed(42)
    n = 200
    
    data = {
        'white_rating': np.random.normal(1500, 200, n),
        'black_rating': np.random.normal(1500, 200, n),
        'mean_risk': np.random.uniform(0, 100, n),
        'event': np.random.choice([' b ', ' c '], n), # Simulate raw event strings with spaces
        
        # Ensure we have all 3 types of outcomes represented
        'winner': np.random.choice(['white', 'black', 'draw'], n),
        'termination_reason': np.random.choice(['normal', 'Time forfeit'], n) # Note capitalization
    }
    
    df = pd.DataFrame(data)
    
    # Inject a known logic case for manual verification
    # Case 1: Draw (Should be 'Draw')
    df.iloc[0] = [1500, 1500, 50, ' b ', 'draw', 'normal']
    # Case 2: Time Forfeit (Should be 'Time Forfeit')
    df.iloc[1] = [1500, 1500, 50, ' b ', 'white', 'Time forfeit']
    # Case 3: Normal Win (Should be 'Normal Decisive')
    df.iloc[2] = [1500, 1500, 50, ' b ', 'black', 'normal']
    
    # Inject a missing value to ensure cleaning works
    df.iloc[3, 0] = np.nan # Missing white_rating
    
    return df

# TESTS

def test_outcome_categorization_logic(mock_chess_data):
    """
    Verifies that the logic converting winner/termination to 
    'game_outcome' is 100% correct.
    """
    df = mock_chess_data.copy()
    
    # Apply your logic (Function approach as used in main script)
    def categorize_outcome(row):
        if row['winner'] == 'draw':
            return 'Draw'
        elif row['termination_reason'] == 'Time forfeit':
            return 'Time Forfeit'
        else:
            return 'Normal Decisive'

    df['game_outcome'] = df.apply(categorize_outcome, axis=1)
    
    # Assert specific rows we manually set in the fixture
    assert df.iloc[0]['game_outcome'] == 'Draw', "Logic Error: Draw winner didn't result in 'Draw'"
    assert df.iloc[1]['game_outcome'] == 'Time Forfeit', "Logic Error: Time forfeit didn't result in 'Time Forfeit'"
    assert df.iloc[2]['game_outcome'] == 'Normal Decisive', "Logic Error: Normal win didn't result in 'Normal Decisive'"

def test_data_preparation_pipeline(mock_chess_data):
    """
    Verifies the full data prep pipeline: cleaning, mapping, and feature engineering.
    """
    df = mock_chess_data.copy()
    
    # 1. Clean Event & Map Time Control
    df['event'] = df['event'].str.strip()
    event_map = {'b': 'Blitz', 'c': 'Classical'}
    df['time_control'] = df['event'].map(event_map)
    
    # Assert mapping worked
    assert set(df['time_control'].unique()) == {'Blitz', 'Classical'}, "Time control mapping failed"
    
    # 2. Calculate Avg Rating
    df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2
    assert not df['avg_rating'].isna().all(), "Avg rating calculation failed"

def test_binary_logistic_regression_model(mock_chess_data):
    """
    Verifies that the Binary Logistic Regression (Draw vs Normal Decisive) runs correctly.
    """
    # Setup Data
    df = mock_chess_data.copy()
    df['event'] = df['event'].str.strip()
    event_map = {'b': 'Blitz', 'c': 'Classical'}
    df['time_control'] = df['event'].map(event_map)
    
    def categorize_outcome(row):
        if row['winner'] == 'draw':
            return 'Draw'
        elif row['termination_reason'] == 'Time forfeit':
            return 'Time Forfeit'
        else:
            return 'Normal Decisive'
    df['game_outcome'] = df.apply(categorize_outcome, axis=1)
    
    df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2
    
    # Filter for Played Games (Exclude Forfeits)
    df_played = df[df['game_outcome'] != 'Time Forfeit'].copy()
    df_played = df_played.dropna() # Drop the injected NaN
    
    # Create Binary Target
    df_played['draw_binary'] = (df_played['game_outcome'] == 'Draw').astype(int)
    
    # Standardize
    df_played['mean_risk_z'] = (df_played['mean_risk'] - df_played['mean_risk'].mean()) / df_played['mean_risk'].std()
    df_played['avg_rating_z'] = (df_played['avg_rating'] - df_played['avg_rating'].mean()) / df_played['avg_rating'].std()
    
    # Fit Model (Blitz subset)
    subset = df_played[df_played['time_control'] == 'Blitz']
    formula = 'draw_binary ~ mean_risk_z + avg_rating_z'
    
    try:
        logit_model = smf.logit(formula=formula, data=subset).fit(disp=0)
    except Exception as e:
        pytest.fail(f"Binary Logit Model fitting crashed: {str(e)}")
        
    # Assertions
    assert 'mean_risk_z' in logit_model.params.index, "Risk variable missing from model"
    assert 'avg_rating_z' in logit_model.params.index, "Rating variable missing from model"
    assert not logit_model.pvalues.isna().all(), "Model failed to converge (NaN p-values)"

def test_time_forfeit_model(mock_chess_data):
    """
    Verifies that the Time Forfeit Logistic Regression runs correctly.
    """
    # Setup Data
    df = mock_chess_data.copy()
    df['event'] = df['event'].str.strip()
    event_map = {'b': 'Blitz', 'c': 'Classical'}
    df['time_control'] = df['event'].map(event_map)
    
    def categorize_outcome(row):
        if row['winner'] == 'draw':
            return 'Draw'
        elif row['termination_reason'] == 'Time forfeit':
            return 'Time Forfeit'
        else:
            return 'Normal Decisive'
    df['game_outcome'] = df.apply(categorize_outcome, axis=1)
    
    df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2
    df['rating_diff'] = abs(df['white_rating'] - df['black_rating'])
    
    df = df.dropna()
    
    # Create Binary Target
    df['time_forfeit'] = (df['game_outcome'] == 'Time Forfeit').astype(int)
    
    # Standardize
    df['mean_risk_z'] = (df['mean_risk'] - df['mean_risk'].mean()) / df['mean_risk'].std()
    df['avg_rating_z'] = (df['avg_rating'] - df['avg_rating'].mean()) / df['avg_rating'].std()
    df['rating_diff_z'] = (df['rating_diff'] - df['rating_diff'].mean()) / df['rating_diff'].std()
    
    # Fit Model
    formula = 'time_forfeit ~ C(time_control) + avg_rating_z + rating_diff_z + mean_risk_z'
    
    try:
        forfeit_model = smf.logit(formula=formula, data=df).fit(disp=0)
    except Exception as e:
        pytest.fail(f"Time Forfeit Model fitting crashed: {str(e)}")
        
    # Assertions
    assert 'mean_risk_z' in forfeit_model.params.index, "Risk variable missing from forfeit model"
    assert 'C(time_control)[T.Classical]' in forfeit_model.params.index, "Time control dummy missing"