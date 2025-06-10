"""
RAPM (Regularized Adjusted Plus-Minus) Data Preprocessing Pipeline
================================================================

This module processes raw soccer event data into a format suitable for RAPM modeling.
It handles lineup extraction, stint creation, and feature engineering for player impact analysis.

Key Features:
- Smart team name extraction from game strings
- Proper stint boundaries (substitutions, goals, red cards, half-time)
- One-hot encoding for offensive/defensive lineups
- Data validation and integrity checks


"""

import pandas as pd
import numpy as np
import ast
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAPMPreprocessor:
    """
    Main class for RAPM data preprocessing pipeline.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RAPM preprocessor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or self._get_default_config()
        self.mlb = MultiLabelBinarizer()
        

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration parameters."""
        project_root = Path(__file__).parent.parent.parent  # Go up to project root
        
        return {
            'min_players_per_team': 6,
            'min_stint_duration': 0.5,
            'default_match_end': 93,
            'events_file': project_root / 'data' / 'raw' / 'pbp-data-soccer' / 'checkbulilineupstotalwith2025.csv',
            'name_mapping_file': project_root / 'data' / 'raw' / 'pbp-data-soccer' / 'namemapping.csv',
            'output_file': project_root / 'data' / 'processed' / 'rapm' / 'rapm_input_clean.pkl'
        }
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load raw events data and apply initial cleaning.
        
        Returns:
            Cleaned events DataFrame
        """
        logger.info("Loading and cleaning raw events data...")
        
        # Load events data
        events = pd.read_csv(self.config['events_file'])
        logger.info(f"Loaded {len(events)} events from {self.config['events_file']}")
        
        # Extract team information
        events = self._extract_team_info(events)
        
        # Apply name mapping
        events = self._apply_name_mapping(events)
        
        # Parse lineup columns
        events = self._parse_lineups(events)
        
        # Clean time information
        events = self._clean_time_data(events)
        
        # Filter valid lineups
        events = self._filter_valid_lineups(events)
        
        logger.info(f"Cleaned data: {len(events)} valid events remaining")
        return events
    
    def _extract_team_info(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract home and away team information from game strings."""
        
        def extract_teams(row):
            """Smart team extraction handling various game string formats."""
            # Remove date prefix
            teams_str = row['game'].split(' ', 1)[1]
            
            # Try different split strategies because some teams have hyphens in their name
            left1, right1 = teams_str.split('-', 1)     # First hyphen
            left2, right2 = teams_str.rsplit('-', 1)    # Last hyphen
            
            team = row['team'].strip()
            
            # Match against actual team name
            if team in [left2.strip(), right2.strip()]:
                home, away = left2.strip(), right2.strip()
            elif team in [left1.strip(), right1.strip()]:
                home, away = left1.strip(), right1.strip()
            else:
                home, away = left2.strip(), right2.strip()
            
            return pd.Series({'home_team': home, 'away_team': away})
        
        teams_info = events.apply(extract_teams, axis=1)
        return pd.concat([events, teams_info], axis=1)
    
    def _apply_name_mapping(self, events: pd.DataFrame) -> pd.DataFrame:
        """Apply standardized name mapping if mapping file exists."""
        try:
            map_df = pd.read_csv(self.config['name_mapping_file'])
            name_map = dict(zip(map_df['raw_name'], map_df['standard_name']))
            
            for col in ['team', 'home_team', 'away_team']:
                events[col] = events[col].replace(name_map)
            
            logger.info(f"Applied name mapping for {len(name_map)} entities")
        except FileNotFoundError:
            logger.warning(f"Name mapping file {self.config['name_mapping_file']} not found, skipping...")
        
        return events
    
    def _parse_lineups(self, events: pd.DataFrame) -> pd.DataFrame:
        """Convert lineup strings to proper lists."""
        
        def to_list(x):
            """Convert various lineup formats to list."""
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return [p.strip() for p in x.split(',') if p.strip()]
            return []
        
        events['home_lineup'] = events['home_lineup'].apply(to_list)
        events['away_lineup'] = events['away_lineup'].apply(to_list)
        
        return events
    
    def _clean_time_data(self, events: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize time information."""
        
        def assign_half(minute_str: str) -> int:
            """Assign half based on minute string."""
            s = str(minute_str).strip()
            if s.startswith("45+"):
                return 1
            try:
                base = int(s.split("+", 1)[0])
                return 1 if base <= 45 else 2
            except ValueError:
                return np.nan
        
        def clean_minute(minute_str: str) -> float:
            """Convert minute strings to numeric values."""
            s = str(minute_str).strip()
            if "+" in s:
                try:
                    base, extra = s.split("+", 1)
                    return int(base) + int(extra)
                except ValueError:
                    return np.nan
            else:
                try:
                    return int(s)
                except ValueError:
                    return np.nan
        
        events['half'] = events['minute'].apply(assign_half)
        events['minute_clean'] = events['minute'].apply(clean_minute)
        
        # Sort by game time
        events = events.sort_values(['game_id', 'half', 'minute_clean']).reset_index(drop=True)
        
        return events
    
    def _filter_valid_lineups(self, events: pd.DataFrame) -> pd.DataFrame:
        """Filter events with valid lineup information."""
        min_players = self.config['min_players_per_team']
        
        valid_mask = (
            events['home_lineup'].apply(lambda x: isinstance(x, list) and len(x) >= min_players) &
            events['away_lineup'].apply(lambda x: isinstance(x, list) and len(x) >= min_players)
        )
        
        filtered_events = events[valid_mask].reset_index(drop=True)
        logger.info(f"Filtered out {len(events) - len(filtered_events)} events with insufficient lineup data")
        
        return filtered_events
    
    def create_stints(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Create RAPM stints from events data.
        
        Args:
            events: Cleaned events DataFrame
            
        Returns:
            DataFrame with stint-level information
        """
        logger.info("Creating RAPM stints...")
        
        summaries = []
        games_processed = 0
        
        for game_id in events['game_id'].unique():
            game_events = events[events['game_id'] == game_id].copy().reset_index(drop=True)
            game_summaries = self._process_game_stints(game_events)
            summaries.extend(game_summaries)
            games_processed += 1
            
            if games_processed % 100 == 0:
                logger.info(f"Processed {games_processed} games...")
        
        stint_df = pd.DataFrame(summaries)
        logger.info(f"Created {len(stint_df)} stints from {games_processed} games")
        
        return stint_df
    
    def _process_game_stints(self, game_events: pd.DataFrame) -> List[Dict]:
        """Process stints for a single game."""
        summaries = []
        
        # Initialize game state
        current_stint = 1
        stint_start_minute = 0
        current_home_lineup = game_events.iloc[0]['home_lineup'].copy()
        current_away_lineup = game_events.iloc[0]['away_lineup'].copy()
        current_stint_events = []
        
        # Calculate game boundaries
        half_time_break = self._calculate_half_time_break(game_events)
        match_end = self._calculate_match_end(game_events)
        
        # Process each event
        for i, row in game_events.iterrows():
            current_minute = row['minute_clean']
            current_stint_events.append(row)
            
            # Update lineups for substitutions and red cards
            self._update_lineups(row, current_home_lineup, current_away_lineup)
            
            # Check for stint-ending conditions
            should_end_stint, reason = self._check_stint_end_conditions(
                row, current_minute, stint_start_minute, half_time_break
            )
            
            is_last_event = (i == len(game_events) - 1)
            
            if should_end_stint or is_last_event:
                stint_end_minute = match_end if (is_last_event and not should_end_stint) else current_minute
                if is_last_event and not should_end_stint:
                    reason = 'match_end'
                
                # Create stint summary
                stint_summary = self._create_stint_summary(
                    game_events.iloc[0], current_stint, stint_start_minute, stint_end_minute,
                    current_home_lineup.copy(), current_away_lineup.copy(),
                    current_stint_events, reason
                )
                summaries.append(stint_summary)
                
                # Initialize next stint
                if not is_last_event:
                    current_stint += 1
                    stint_start_minute = stint_end_minute
                    current_stint_events = []
        
        return summaries
    
    def _calculate_half_time_break(self, game_events: pd.DataFrame) -> float:
        """Calculate half-time break minute."""
        first_half_events = game_events[game_events['half'] == 1]
        if len(first_half_events) > 0:
            return first_half_events['minute_clean'].max() + 0.1
        return 45.1
    
    def _calculate_match_end(self, game_events: pd.DataFrame) -> float:
        """Calculate realistic match end time."""
        second_half_events = game_events[game_events['half'] == 2]
        if len(second_half_events) > 0:
            last_minute = second_half_events['minute_clean'].max()
            return max(last_minute + 2, self.config['default_match_end'])
        return self.config['default_match_end']
    
    def _update_lineups(self, row: pd.Series, home_lineup: List[str], away_lineup: List[str]) -> None:
        """Update lineups based on substitutions and red cards."""
        # Handle substitutions
        if row['event_type'] == 'substitute_in':
            side = 'home' if row['team'] == row['home_team'] else 'away'
            out_player, in_player = row['player2'], row['player1']
            
            target_lineup = home_lineup if side == 'home' else away_lineup
            if out_player in target_lineup:
                target_lineup[target_lineup.index(out_player)] = in_player
        
        # Handle red cards
        elif self._is_red_card_event(row):
            side = 'home' if row['team'] == row['home_team'] else 'away'
            red_carded_player = row['player1']
            
            target_lineup = home_lineup if side == 'home' else away_lineup
            if red_carded_player in target_lineup:
                target_lineup.remove(red_carded_player)
                logger.debug(f"Red card: {red_carded_player} removed from {row['team']}")
    
    def _is_red_card_event(self, row: pd.Series) -> bool:
        """Check if event is a red card."""
        event_type = str(row.get('event_type', '')).lower()
        return 'red_card' in event_type or 'yellow_red_card' in event_type
    
    def _check_stint_end_conditions(self, row: pd.Series, current_minute: float, 
                                  stint_start: float, half_time_break: float) -> Tuple[bool, str]:
        """Check if stint should end based on game events."""
        if row['event_type'] == 'substitute_in':
            return True, 'substitution'
        elif row['outcome'] == 'Goal':
            return True, 'goal'
        elif self._is_red_card_event(row):
            return True, 'red_card'
        elif current_minute >= half_time_break and stint_start < half_time_break:
            return True, 'half_time'
        
        return False, None
    
    def _create_stint_summary(self, game_info: pd.Series, stint_num: int, 
                            start_min: float, end_min: float,
                            home_lineup: List[str], away_lineup: List[str],
                            stint_events: List[pd.Series], reason: str) -> Dict:
        """Create summary dictionary for a stint."""
        duration = max(end_min - start_min, self.config['min_stint_duration'])
        
        # Calculate xG statistics
        stint_df = pd.DataFrame(stint_events)
        home_xg = stint_df[stint_df['team'] == game_info['home_team']]['xG'].fillna(0).sum()
        away_xg = stint_df[stint_df['team'] == game_info['away_team']]['xG'].fillna(0).sum()
        
        return {
            'game_id': game_info['game_id'],
            'stint': stint_num,
            'minutes_played': duration,
            'home_lineup': home_lineup,
            'away_lineup': away_lineup,
            'home_xG': home_xg,
            'away_xG': away_xg,
            'total_xG': home_xg + away_xg,
            'stint_reason': reason,
            'league': game_info['league'],
            'season': game_info['season'],
            'game': game_info['game'],
            'home_team': game_info['home_team'],
            'away_team': game_info['away_team'],
            'stint_start': start_min,
            'stint_end': end_min,
            'num_events': len(stint_events),
            'home_players': len(home_lineup),
            'away_players': len(away_lineup)
        }
    
    def create_rapm_features(self, stint_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform stint data into RAPM modeling format.
        
        Args:
            stint_df: DataFrame with stint information
            
        Returns:
            RAPM-ready DataFrame with one-hot encoded features
        """
        logger.info("Creating RAPM feature matrix...")
        
        # Convert lineup strings to lists if needed
        stint_df = self._ensure_lineup_lists(stint_df)
        
        # Create home/away perspective rows
        rapm_df = self._create_perspective_rows(stint_df)
        
        # One-hot encode lineups
        rapm_df = self._encode_lineups(rapm_df)
        
        # Set target variable
        rapm_df = self._set_target_variable(rapm_df)
        
        logger.info(f"Created RAPM feature matrix: {rapm_df.shape}")
        return rapm_df
    
    def _ensure_lineup_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure lineup columns are proper lists."""
        for col in ['home_lineup', 'away_lineup']:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(ast.literal_eval)
        return df
    
    def _create_perspective_rows(self, stint_df: pd.DataFrame) -> pd.DataFrame:
        """Create separate rows for home/away offensive perspectives."""
        # Home team perspective (home_flag=1)
        home_perspective = stint_df.copy()
        home_perspective['home_flag'] = 1
        home_perspective['offense_list'] = home_perspective['home_lineup']
        home_perspective['defense_list'] = home_perspective['away_lineup']
        
        # Away team perspective (home_flag=0)
        away_perspective = stint_df.copy()
        away_perspective['home_flag'] = 0
        away_perspective['offense_list'] = away_perspective['away_lineup']
        away_perspective['defense_list'] = away_perspective['home_lineup']
        
        return pd.concat([home_perspective, away_perspective], ignore_index=True)
    
    def _encode_lineups(self, rapm_df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode offensive and defensive lineups."""
        # Fit on offensive lineups
        offense_features = pd.DataFrame(
            self.mlb.fit_transform(rapm_df['offense_list']),
            columns=[f"{player}_offense" for player in self.mlb.classes_],
            index=rapm_df.index
        )
        
        # Transform defensive lineups with same encoding
        defense_features = pd.DataFrame(
            self.mlb.transform(rapm_df['defense_list']),
            columns=[f"{player}_defense" for player in self.mlb.classes_],
            index=rapm_df.index
        )
        
        # Combine features
        feature_df = pd.concat([
            rapm_df.drop(columns=['home_lineup', 'away_lineup', 'offense_list', 'defense_list']),
            offense_features,
            defense_features
        ], axis=1)
        
        logger.info(f"Created {len(offense_features.columns)} offense and {len(defense_features.columns)} defense features")
        return feature_df
    
    def _set_target_variable(self, rapm_df: pd.DataFrame) -> pd.DataFrame:
        """Set target variable based on offensive perspective."""
        rapm_df['total_xG'] = np.where(
            rapm_df['home_flag'] == 1,
            rapm_df['home_xG'],
            rapm_df['away_xG']
        )
        return rapm_df
    
    def validate_data_integrity(self, events: pd.DataFrame, stint_df: pd.DataFrame) -> None:
        """Validate that data processing preserved integrity."""
        logger.info("Validating data integrity...")
        
        # Check xG preservation
        total_events_xg = events['xG'].fillna(0).sum()
        total_stint_xg = stint_df['total_xG'].sum()
        xg_diff = abs(total_events_xg - total_stint_xg)
        
        if xg_diff > 0.001:
            logger.warning(f"xG mismatch detected: {xg_diff:.6f}")
        else:
            logger.info("✅ xG values preserved correctly")
        
        # Check stint statistics
        avg_duration = stint_df['minutes_played'].mean()
        logger.info(f"Average stint duration: {avg_duration:.1f} minutes")
        
        # Check lineup sizes
        min_home_players = stint_df['home_players'].min()
        min_away_players = stint_df['away_players'].min()
        logger.info(f"Minimum players: Home={min_home_players}, Away={min_away_players}")
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete RAPM preprocessing pipeline.
        
        Returns:
            Final RAPM-ready DataFrame
        """
        logger.info("Starting RAPM preprocessing pipeline...")
        
        # Step 1: Load and clean raw data
        events = self.load_and_clean_data()
        
        # Step 2: Create stints
        stint_df = self.create_stints(events)
        
        # Step 3: Validate data integrity
        self.validate_data_integrity(events, stint_df)
        
        # Step 4: Create RAPM features
        rapm_df = self.create_rapm_features(stint_df)
        
        # Step 5: Save results
        rapm_df.to_pickle(self.config['output_file'])
        logger.info(f"✅ Saved final RAPM data to {self.config['output_file']}")
        
        return rapm_df


def main():
    """Main execution function."""
    # Initialize preprocessor
    preprocessor = RAPMPreprocessor()
    
    # Run full pipeline
    rapm_data = preprocessor.run_full_pipeline()
    
    print(f"\n=== RAPM PREPROCESSING COMPLETE ===")
    print(f"Final dataset shape: {rapm_data.shape}")
    print(f"Target variable range: {rapm_data['total_xG'].min():.3f} to {rapm_data['total_xG'].max():.3f}")
    print(f"Average stint duration: {rapm_data['minutes_played'].mean():.1f} minutes")


if __name__ == "__main__":
    main()