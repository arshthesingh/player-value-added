# preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Set, Optional, Union
import os

def load_data(rapm_path: str = 'data/rapm_input.pkl', 
              gk_path: str = 'data/top5_standard_24.csv',
              player_mins_path: str = 'data/playermins.1724.csv',
              base_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary data files for RAPM processing.
    
    Parameters:
    -----------
    rapm_path : str
        Path to the RAPM input pickle file
    gk_path : str
        Path to goalkeeper data CSV
    player_mins_path : str
        Path to player minutes CSV
    base_dir : str, optional
        Base directory for data files, used if specific paths not provided
        
    Returns:
    --------
    Tuple containing:
        - df: Main RAPM dataframe
        - gk: Goalkeeper dataframe
        - player_mins: Player minutes dataframe
    """
        # If specific paths aren't provided but base_dir is, construct paths
    if base_dir is not None:
        if rapm_path is None:
            rapm_path = os.path.join(base_dir, 'rapm_input.pkl')
        if gk_path is None:
            gk_path = os.path.join(base_dir, 'top5_standard_24.csv')
        if player_mins_path is None:
            player_mins_path = os.path.join(base_dir, 'playermins.1724.csv')
    
        # Default paths if nothing is provided
    else:
        if rapm_path is None:
            rapm_path = 'rapm_input.pkl'
        if gk_path is None:
            gk_path = 'top5_standard_24.csv'
        if player_mins_path is None:
            player_mins_path = 'playermins.1724.csv'

    try:
        df = pd.read_pickle(rapm_path)
        print(f"Loaded RAPM data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading RAPM data: {e}")
        df = None
        
    try:
        gk = pd.read_csv(gk_path)
        print(f"Loaded goalkeeper data with shape: {gk.shape}")
    except Exception as e:
        print(f"Error loading goalkeeper data: {e}")
        gk = None
    
    try:
        player_mins = pd.read_csv(player_mins_path)
        print(f"Loaded player minutes data with shape: {player_mins.shape}")
    except Exception as e:
        print(f"Error loading player minutes data: {e}")
        player_mins = None
    
    return df, gk, player_mins

def process_goalkeeper_data(gk: pd.DataFrame) -> List[str]:
    """
    Process goalkeeper data to identify goalkeeper players.
    
    Parameters:
    -----------
    gk : pd.DataFrame
        Goalkeeper dataframe
        
    Returns:
    --------
    List[str]
        List of goalkeeper player names
    """

    gk['Min_Playing'] = pd.to_numeric(gk['Min_Playing'], errors='coerce')
    
    # Group by player and combine positions
    gk_grouped = (gk
                 .groupby(['Player', 'Url'], as_index=False)
                 .agg(
                     Min_Playing = ('Min_Playing', 'sum'),
                     Pos = ('Pos', lambda x: ', '.join(sorted(x.dropna().astype(str).unique())))))
    
    # Extract goalkeeper players
    gk_players = gk_grouped.loc[gk_grouped['Pos'].str.contains('GK', na=False), 'Player'].tolist()
    print(f"Identified {len(gk_players)} goalkeeper players to remove")
    
    return gk_players

def identify_replacement_players(player_mins: pd.DataFrame, threshold: int = 2000) -> Set[str]:
    """
    Identify replacement-level players based on minutes threshold.
    
    Parameters:
    -----------
    player_mins : pd.DataFrame
        Player minutes dataframe
    threshold : int
        Minute threshold for replacement-level players
        
    Returns:
    --------
    Set[str]
        Set of replacement-level player names
    """
    replacement_players = set(player_mins.loc[player_mins['Min_Playing'] < threshold, 'Player'])
    print(f"Found {len(replacement_players)} replacement-level players (threshold: {threshold} minutes)")
    
    return replacement_players

def preprocess_rapm_data(df: pd.DataFrame = None, 
                        gk: pd.DataFrame = None, 
                        player_mins: pd.DataFrame = None,
                        rapm_path: str = 'data/rapm_input.pkl',
                        gk_path: str = 'data/top5_standard_24.csv',
                        player_mins_path: str = 'data/playermins.1724.csv',
                        base_dir: str = None,
                        replacement_threshold: int = 2000,
                        include_xg_per_90: bool = True) -> pd.DataFrame:
    """
    Comprehensive preprocessing for RAPM data:
    1. Load data if not provided
    2. Remove goalkeeper players
    3. Combine replacement-level players
    4. Calculate target (e.g., xG per 90)
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Pre-loaded RAPM dataframe
    gk : pd.DataFrame, optional
        Pre-loaded goalkeeper dataframe
    player_mins : pd.DataFrame, optional
        Pre-loaded player minutes dataframe
    rapm_path : str
        Path to the RAPM input pickle file (used if df not provided)
    gk_path : str
        Path to goalkeeper data CSV (used if gk not provided)
    player_mins_path : str
        Path to player minutes CSV (used if player_mins not provided)
    base_dir : str, optional
        Base directory for data files, used if specific paths not provided
    replacement_threshold : int
        Minute threshold for replacement-level players
    include_xg_per_90 : bool
        Whether to calculate xG per 90 minutes
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe ready for RAPM modeling
    """
    # Load data if not provided
    if df is None or gk is None or player_mins is None:
        df, gk, player_mins = load_data(rapm_path, gk_path, player_mins_path, base_dir)
    
    if df is None or gk is None or player_mins is None:
        raise ValueError("Failed to load required data files")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Get list of goalkeeper players
    gk_players = process_goalkeeper_data(gk)
    
    # Remove goalkeeper columns
    off_cols = [f"{p}_offense" for p in gk_players if f"{p}_offense" in df.columns]
    def_cols = [f"{p}_defense" for p in gk_players if f"{p}_defense" in df.columns]
    to_drop = off_cols + def_cols 
    print(f"Dropping {len(to_drop)} goalkeeper columns")
    
    df = df.drop(columns=to_drop, errors='ignore')
    
    # Handle replacement-level players
    replacement_players = identify_replacement_players(player_mins, replacement_threshold)
    
    off_cols = [f"{p}_offense" for p in replacement_players if f"{p}_offense" in df.columns]
    def_cols = [f"{p}_defense" for p in replacement_players if f"{p}_defense" in df.columns]
    
    # Create replacement player indicators
    df['replacement_offense'] = (df[off_cols] == 1).any(axis=1).astype(int)
    df['replacement_defense'] = (df[def_cols] == 1).any(axis=1).astype(int)
    
    # Drop individual replacement player columns
    print(f"Dropping {len(off_cols) + len(def_cols)} replacement player columns")
    df = df.drop(columns=off_cols + def_cols)
    
    # Calculate xG per 90
    if include_xg_per_90 and 'total_xG' in df.columns and 'minutes_played' in df.columns:
        df['xG_per_90'] = (df['total_xG'] / df['minutes_played']) * 90
        print("Added xG_per_90 feature")
    
    print(f"Final preprocessed dataframe shape: {df.shape}")
    return df
