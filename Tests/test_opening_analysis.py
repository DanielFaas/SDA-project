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
        'event': np.random.choice(['blitz', 'classical'], n),
        
        # Ensure we have all 3 types of outcomes represented
        'winner': np.random.choice(['white', 'black', 'draw'], n),
        'termination_reason': np.random.choice(['normal', 'time forfeit'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Inject a known logic case for manual verification
    # Case 1: Draw (Should be 'Draw')
    df.iloc[0] = [1500, 1500, 50, 'blitz', 'draw', 'normal']
    # Case 2: Time Forfeit (Should be 'Time Forfeit')
    df.iloc[1] = [1500, 1500, 50, 'blitz', 'white', 'time forfeit']
    # Case 3: Normal Win (Should be 'Normal Decisive')
    df.iloc[2] = [1500, 1500, 50, 'blitz', 'black', 'normal']
    
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
    
    # Apply your logic (Vectorized approach)
    conditions = [
        (df['winner'] == 'draw'),
        (df['termination_reason'] == 'time forfeit')
    ]
    choices = ['Draw', 'Time Forfeit']
    df['game_outcome'] = np.select(conditions, choices, default='Normal Decisive')
    
    # Assert specific rows we manually set in the fixture
    assert df.iloc[0]['game_outcome'] == 'Draw', "Logic Error: Draw winner didn't result in 'Draw'"
    assert df.iloc[1]['game_outcome'] == 'Time Forfeit', "Logic Error: Time forfeit didn't result in 'Time Forfeit'"
    assert df.iloc[2]['game_outcome'] == 'Normal Decisive', "Logic Error: Normal win didn't result in 'Normal Decisive'"

def test_data_cleaning_and_preparation(mock_chess_data):
    """
    Verifies that NaNs are dropped and categorical conversion works.
    """
    df = mock_chess_data.copy()
    
    # Pre-check: we put a NaN in row 3
    assert df['white_rating'].isna().sum() > 0
    
    # Apply Cleaning
    df_clean = df.dropna(subset=['mean_risk', 'white_rating', 'black_rating', 'event'])
    
    # Assert NaNs are gone
    assert df_clean.shape[0] < df.shape[0], "Failed to drop rows with missing values"
    assert df_clean['white_rating'].isna().sum() == 0

def test_regression_model_runs_successfully(mock_chess_data):
    """
    The Big Test: Does the statistical model actually fit without crashing?
    """
    # 1. Setup
    df = mock_chess_data.copy()
    
    # Apply Logic
    conditions = [
        (df['winner'] == 'draw'),
        (df['termination_reason'] == 'time forfeit')
    ]
    choices = ['Draw', 'Time Forfeit']
    df['game_outcome'] = np.select(conditions, choices, default='Normal Decisive')
    
    # Clean
    df_clean = df.dropna().copy()
    df_clean['avg_rating'] = (df_clean['white_rating'] + df_clean['black_rating']) / 2
    
    # 2. Prepare Categories (The critical fix)
    df_clean['game_outcome'] = pd.Categorical(
        df_clean['game_outcome'], 
        categories=['Draw', 'Normal Decisive', 'Time Forfeit'], 
        ordered=True
    )
    df_clean['outcome_code'] = df_clean['game_outcome'].cat.codes
    
    # 3. Fit Model
    model_formula = 'outcome_code ~ mean_risk + avg_rating + C(event)'
    
    try:
        mnlogit_model = smf.mnlogit(formula=model_formula, data=df_clean).fit(disp=0)
    except Exception as e:
        pytest.fail(f"Model fitting crashed with error: {str(e)}")
        
    # 4. Assertions on Results
    # params should have shape (Num_Features, Num_Outcomes - 1)
    # We have 2 outcomes (Normal, Forfeit) predicted relative to Reference (Draw)
    # We have Intercept + risk + rating + event(T.c) = 4 features
    
    assert mnlogit_model.params.shape[1] == 2, "Model should predict 2 outcomes (relative to Draw)"
    
    # Check if p-values exist (meaning convergence worked)
    assert not mnlogit_model.pvalues.isna().all().all(), "Model returned all NaN p-values (did not converge)"
    
    # Check if 'mean_risk' is actually in the model
    assert 'mean_risk' in mnlogit_model.params.index, "mean_risk variable was lost in the model"