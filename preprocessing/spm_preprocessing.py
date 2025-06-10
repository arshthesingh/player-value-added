"""
SPM (Statistical Plus-Minus) Data Preprocessing Pipeline
======================================================

This module processes FBRef player and team statistics to create features for SPM modeling.
It handles data merging, per-90 calculations, possession adjustments, and advanced metrics.

Key Features:
- Multi-table data merging with suffix handling
- Per-90 minute rate calculations
- Possession-adjusted defensive metrics
- Touch centrality and advanced possession metrics
- Team-level context integration


"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import reduce

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SPMPreprocessor:
    """
    Main class for SPM data preprocessing pipeline.
    Handles FBRef player and team data integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SPM preprocessor.
        
        Args:
            config: Configuration dictionary with file paths and parameters
        """
        self.config = config or self._get_default_config()
        self.player_data_files = self._get_player_data_files()
        self.team_data_files = self._get_team_data_files()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration parameters."""
        project_root = Path(__file__).parent.parent.parent
        
        return {
            'data_dir': project_root / 'data' / 'raw' / 'fbref',
            'output_file': project_root / 'data' / 'processed' / 'spm' / 'cleaned_fbref_players.csv',
            'season': '25',  # Season identifier
            'per90_start_col': 'O',  # Excel column for per-90 calculations start
            'per90_mid_col': 'AC',   # Excel column for per-90 calculations middle
            'per90_end_col': 'BJ',   # Excel column for per-90 calculations end
            'possession_adjustment_baseline': 50  # Baseline for possession adjustments
        }
    
    def _get_player_data_files(self) -> Dict[str, str]:
        """Define player data file mappings."""
        season = self.config['season']
        return {
            'standard': f'top5_standard_{season}.csv',
            'playing_time': f'top5_playing_time_{season}.csv',
            'possession': f'top5_possession_{season}.csv',
            'defense': f'top5_defense_{season}.csv',
            'gca': f'top5_gca_{season}.csv',
            'pass_types': f'top5_passtypes_{season}.csv',
            'passing': f'top5_passing_{season}.csv',
            'misc': f'top5_misc_{season}.csv'
        }
    
    def _get_team_data_files(self) -> Dict[str, str]:
        """Define team data file mappings."""
        season = self.config['season']
        return {
            'standard': f'top5_standard_{season}_team.csv',
            'possession': f'top5_possession_{season}_team.csv'
        }
    
    def load_player_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all player data files.
        
        Returns:
            Dictionary of DataFrames keyed by data type
        """
        logger.info("Loading player data files...")
        
        player_dfs = {}
        data_dir = self.config['data_dir']
        
        for data_type, filename in self.player_data_files.items():
            file_path = data_dir / filename
            try:
                df = pd.read_csv(file_path)
                player_dfs[data_type] = df
                logger.info(f"Loaded {len(df)} rows from {filename}")
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
        
        return player_dfs
    
    def load_team_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load team data files.
        
        Returns:
            Dictionary of team DataFrames
        """
        logger.info("Loading team data files...")
        
        team_dfs = {}
        data_dir = self.config['data_dir']
        
        for data_type, filename in self.team_data_files.items():
            file_path = data_dir / filename
            try:
                df = pd.read_csv(file_path)
                team_dfs[data_type] = df
                logger.info(f"Loaded {len(df)} rows from {filename}")
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
        
        return team_dfs
    
    def merge_player_data(self, player_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all player data files on common keys.
        
        Args:
            player_dfs: Dictionary of player DataFrames
            
        Returns:
            Merged player DataFrame
        """
        logger.info("Merging player data files...")
        
        # Get DataFrames in merge order (standard first)
        merge_order = ['standard', 'playing_time', 'possession', 'defense', 
                      'gca', 'pass_types', 'passing', 'misc']
        
        dfs_to_merge = [player_dfs[key] for key in merge_order if key in player_dfs]
        
        def merge_left(base: pd.DataFrame, other: pd.DataFrame, idx: int) -> pd.DataFrame:
            """Merge two DataFrames with suffix handling."""
            # Remove duplicates in joining DataFrame
            other_clean = other.drop_duplicates(subset=['Season_End_Year', 'Squad', 'Player'])
            
            # Perform left join with suffixes
            merged = base.merge(
                other_clean,
                on=['Season_End_Year', 'Squad', 'Player'],
                how='left',
                suffixes=('', f'_df{idx}')
            )
            
            return merged
        
        # Reduce over all DataFrames
        merged = reduce(
            lambda left, right_with_idx: merge_left(left, right_with_idx[1], right_with_idx[0]),
            enumerate(dfs_to_merge[1:], start=2),
            dfs_to_merge[0]
        )
        
        # Validate merge integrity
        base_length = len(dfs_to_merge[0])
        assert len(merged) == base_length, f"Row count changed during merge: {base_length} -> {len(merged)}"
        
        logger.info(f"Successfully merged {len(dfs_to_merge)} player datasets")
        return merged
    
    def clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean merged DataFrame by removing duplicates and unnecessary columns.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning merged data...")
        
        # Remove rows with missing player names
        df = df.dropna(subset=['Player']).reset_index(drop=True)
        
        # Define columns to drop (duplicated metadata from merges)
        cols_to_drop = self._get_columns_to_drop()
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Reorder columns (move URL to front)
        df = self._reorder_columns(df)
        
        # Convert numeric columns
        df.iloc[:, 9:] = df.iloc[:, 9:].apply(pd.to_numeric, errors='coerce')
        
        logger.info(f"Cleaned data: {len(df)} players, {len(df.columns)} columns")
        return df
    
    def _get_columns_to_drop(self) -> List[str]:
        """Define columns to drop during cleaning."""
        drop_patterns = [
            'Unnamed: 0_df', 'Comp_df', 'Nation_df', 'Pos_df', 
            'Age_df', 'Born_df', 'Url_df', 'Mins_Per_90_df'
        ]
        
        cols_to_drop = []
        for pattern in drop_patterns:
            for i in range(2, 9):  # df2 through df8
                cols_to_drop.append(f"{pattern}{i}")
        
        return cols_to_drop
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Move URL column to front."""
        if 'Url' in df.columns:
            cols = list(df.columns)
            cols.insert(0, cols.pop(cols.index('Url')))
            df = df[cols]
        return df
    
    def calculate_per90_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-90 minute metrics for relevant columns.
        
        Args:
            df: Player DataFrame
            
        Returns:
            DataFrame with per-90 metrics added
        """
        logger.info("Calculating per-90 minute metrics...")
        
        # Calculate 90s played
        df_with_90s = df.copy()
        df_with_90s['90s'] = df_with_90s['Min_Playing'] / 90
        
        # Get column indices for per-90 calculations
        per90_cols = self._get_per90_columns(df_with_90s.columns.tolist())
        
        # Calculate per-90 metrics
        per90_metrics = (
            df_with_90s[per90_cols]
            .div(df_with_90s['90s'], axis=0)
            .add_suffix('Per90')
        )
        
        # Join back to original DataFrame
        result = df.join(per90_metrics)
        
        logger.info(f"Added {len(per90_metrics.columns)} per-90 metrics")
        return result
    
    def _get_per90_columns(self, all_columns: List[str]) -> List[str]:
        """Determine which columns should be converted to per-90 metrics."""
        def excel_to_idx(col_letter: str) -> int:
            """Convert Excel column letter to zero-based index."""
            col_letter = col_letter.upper()
            idx = 0
            for char in col_letter:
                idx = idx * 26 + (ord(char) - ord('A') + 1)
            return idx - 1
        
        # Get column slices based on Excel column references
        o_idx = excel_to_idx(self.config['per90_start_col'])
        ac_idx = excel_to_idx(self.config['per90_mid_col'])
        bj_idx = excel_to_idx(self.config['per90_end_col'])
        
        slice1 = all_columns[o_idx:ac_idx+1]
        slice2 = all_columns[bj_idx:]
        
        return slice1 + slice2
    
    def process_team_data(self, team_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process team-level data for player context.
        
        Args:
            team_dfs: Dictionary of team DataFrames
            
        Returns:
            Processed team context DataFrame
        """
        logger.info("Processing team data...")
        
        # Process team standard stats
        df_standard = team_dfs['standard'].copy()
        df_standard = df_standard[df_standard['Team_or_Opponent'] == 'team'].reset_index(drop=True)
        
        # Process team possession stats
        df_poss = team_dfs['possession'].copy()
        df_team_poss = df_poss[df_poss['Team_or_Opponent'] == 'team'].reset_index(drop=True)
        df_opp_poss = df_poss[df_poss['Team_or_Opponent'] == 'opponent'].reset_index(drop=True)
        
        # Create team context DataFrame
        team_context = df_standard.iloc[:, :30].copy()
        
        # Calculate team touches per 90
        team_context['TeamTouches90'] = (
            df_team_poss['Touches_Touches'].astype(float) / 
            df_team_poss['90s'].astype(float)
        )
        
        # Add opponent touches
        team_context['Opp_Touches'] = df_opp_poss['Touches_Touches'].values
        
        # Rename minutes column for clarity
        team_context = team_context.rename(columns={'Min_Playing': 'Team_Min'})
        
        # Convert numeric columns
        team_context.iloc[:, 7:] = team_context.iloc[:, 7:].apply(pd.to_numeric, errors='coerce')
        
        logger.info(f"Processed team context for {len(team_context)} teams")
        return team_context
    
    def add_team_context(self, player_df: pd.DataFrame, team_context: pd.DataFrame) -> pd.DataFrame:
        """
        Add team-level context to player data.
        
        Args:
            player_df: Player DataFrame
            team_context: Team context DataFrame
            
        Returns:
            Player DataFrame with team context
        """
        logger.info("Adding team context to player data...")
        
        # Initialize team context columns
        context_cols = ['AvgTeamPoss', 'OppTouches', 'TeamMins', 'TeamTouches90']
        for col in context_cols:
            player_df[col] = 0.0
        
        # Map team context to players
        for idx, player_row in player_df.iterrows():
            team_name = player_row['Squad']
            team_row = team_context[team_context['Squad'] == team_name]
            
            if not team_row.empty:
                team_row = team_row.iloc[0]
                player_df.at[idx, 'AvgTeamPoss'] = team_row['Poss']
                player_df.at[idx, 'OppTouches'] = team_row['Opp_Touches']
                player_df.at[idx, 'TeamMins'] = team_row['Team_Min']
                player_df.at[idx, 'TeamTouches90'] = team_row['TeamTouches90']
        
        # Ensure numeric types
        numeric_cols = player_df.columns[14:].tolist() + ['Carries_Carries', 'Touches_Touches']
        for col in numeric_cols:
            if col in player_df.columns:
                player_df[col] = pd.to_numeric(player_df[col], errors='coerce')
        
        logger.info("Successfully added team context to player data")
        return player_df
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate possession-adjusted and advanced metrics.
        
        Args:
            df: Player DataFrame with team context
            
        Returns:
            DataFrame with advanced metrics
        """
        logger.info("Calculating advanced metrics...")
        
        baseline = self.config['possession_adjustment_baseline']
        
        # Possession-adjusted defensive metrics
        possession_adjustments = {
            'pAdjTkl+IntPer90': 'Tkl+IntPer90',
            'pAdjClrPer90': 'ClrPer90',
            'pAdjShBlocksPer90': 'Sh_BlocksPer90',
            'pAdjPassBlocksPer90': 'Pass_BlocksPer90',
            'pAdjIntPer90': 'IntPer90',
            'pAdjDrbTklPer90': 'Tkl_ChallengesPer90',
            'pAdjTklWinPossPer90': 'Tkl_ChallengesPer90',
            'pAdjDrbPastPer90': 'Lost_ChallengesPer90',
            'pAdjAerialWinsPer90': 'Won_AerialPer90',
            'pAdjAerialLossPer90': 'Lost_AerialPer90',
            'pAdjDrbPastAttPer90': 'Att_ChallengesPer90'
        }
        
        for new_col, base_col in possession_adjustments.items():
            if base_col in df.columns:
                df[new_col] = (df[base_col] / (100 - df['AvgTeamPoss'])) * baseline
        
        # Touch and possession metrics
        df['TouchCentrality'] = (df['Touches_TouchesPer90'] / df['TeamTouches90']) * 100
        df['pAdjTouchesPer90'] = (df['Touches_TouchesPer90'] / df['AvgTeamPoss']) * baseline
        
        # Efficiency metrics
        df['Tkl+IntPer600OppTouch'] = (
            df['Tkl+Int'] / (df['OppTouches'] * (df['Min_Playing'] / df['TeamMins'])) * 600
        )
        
        # Rate metrics
        df['CarriesPer50Touches'] = self._safe_divide(df['Carries_Carries'], df['Touches_Touches'])
        df['ProgCarriesPer50Touches'] = self._safe_divide(df['PrgC_Carries'], df['Touches_Touches'])
        df['ProgPassesPer50CmpPasses'] = self._safe_divide(df['PrgP'], df['Cmp_Total'])
        
        logger.info(f"Calculated {len(possession_adjustments) + 6} advanced metrics")
        return df
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two series, handling division by zero."""
        return numerator.divide(denominator).fillna(0)
    
    def validate_final_data(self, df: pd.DataFrame) -> None:
        """
        Validate the final processed dataset.
        
        Args:
            df: Final processed DataFrame
        """
        logger.info("Validating final dataset...")
        
        # Check for expected columns
        required_cols = ['Player', 'Squad', 'Min_Playing', 'TouchCentrality', 'pAdjTouchesPer90']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
        
        # Check data quality
        null_percentage = (df.isnull().sum() / len(df) * 100).round(2)
        high_null_cols = null_percentage[null_percentage > 50]
        if not high_null_cols.empty:
            logger.warning(f"Columns with >50% null values: {high_null_cols.to_dict()}")
        
        # Summary statistics
        logger.info(f"Final dataset: {len(df)} players, {len(df.columns)} features")
        logger.info(f"Average minutes played: {df['Min_Playing'].mean():.1f}")
        logger.info(f"Teams represented: {df['Squad'].nunique()}")
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete SPM preprocessing pipeline.
        
        Returns:
            Final processed DataFrame ready for SPM modeling
        """
        logger.info("Starting SPM preprocessing pipeline...")
        
        # Step 1: Load data
        player_dfs = self.load_player_data()
        team_dfs = self.load_team_data()
        
        # Step 2: Merge player data
        merged_players = self.merge_player_data(player_dfs)
        
        # Step 3: Clean merged data
        cleaned_players = self.clean_merged_data(merged_players)
        
        # Step 4: Calculate per-90 metrics
        players_with_per90 = self.calculate_per90_metrics(cleaned_players)
        
        # Step 5: Process team context
        team_context = self.process_team_data(team_dfs)
        
        # Step 6: Add team context to players
        players_with_context = self.add_team_context(players_with_per90, team_context)
        
        # Step 7: Calculate advanced metrics
        final_players = self.calculate_advanced_metrics(players_with_context)
        
        # Step 8: Validate and save
        self.validate_final_data(final_players)
        
        # Ensure output directory exists
        output_path = Path(self.config['output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save final dataset
        final_players.to_csv(self.config['output_file'], index=False)
        logger.info(f"✅ Saved final SPM dataset to {self.config['output_file']}")
        
        return final_players


def main():
    """Main execution function."""
    # Initialize preprocessor
    preprocessor = SPMPreprocessor()
    
    # Run full pipeline
    spm_data = preprocessor.run_full_pipeline()
    
    print(f"\n=== SPM PREPROCESSING COMPLETE ===")
    print(f"Final dataset shape: {spm_data.shape}")
    print(f"Players processed: {len(spm_data)}")
    print(f"Teams represented: {spm_data['Squad'].nunique()}")
    print(f"Average touch centrality: {spm_data['TouchCentrality'].mean():.2f}%")


if __name__ == "__main__":
    main()