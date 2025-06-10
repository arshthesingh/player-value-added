"""
RAPM (Regularized Adjusted Plus-Minus) Demo Model
================================================

This is a simplified RAPM implementation for portfolio demonstration.
The production version includes proprietary weighting schemes, advanced
regularization techniques, and optimized bootstrap procedures.

This demo showcases core RAPM methodology including:
- Ridge regression with cross-validation
- Player impact estimation on team performance
- Bootstrap confidence intervals
- Model validation and interpretation


"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAPMDemo:
    """
    Simplified RAPM (Regularized Adjusted Plus-Minus) model for soccer analytics.
    
    This demo version implements core RAPM methodology without proprietary
    optimizations used in production systems.
    
    RAPM measures individual player impact on team performance by:
    1. Creating one-hot encoded features for players on/off field
    2. Using regularized regression to estimate player effects
    3. Controlling for teammates and opponents through lineup context
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize RAPM demo model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler() if self.config['standardize_features'] else None
        self.results = None
        
    def _get_default_config(self) -> Dict:
        """Default configuration for demo model."""
        return {
            'alpha_range': np.logspace(1, 5, 50),  # Regularization parameter range
            'cv_folds': 5,                         # Cross-validation folds
            'min_minutes_threshold': 500,          # Minimum minutes for player inclusion
            'bootstrap_iterations': 100,           # Simplified bootstrap (production uses more)
            'confidence_level': 0.95,              # Confidence interval level
            'standardize_features': False,         # Feature standardization
            'random_state': 42                     # Reproducibility
        }
    
    def load_and_prepare_data(self, stint_data_path: str) -> pd.DataFrame:
        """
        Load and prepare stint-level data for RAPM modeling.
        
        Args:
            stint_data_path: Path to preprocessed stint data
            
        Returns:
            Cleaned and prepared DataFrame
        """
        logger.info("Loading stint data for RAPM modeling...")
        
        # Load data (expects output from preprocessing pipeline)
        if stint_data_path.endswith('.pkl'):
            data = pd.read_pickle(stint_data_path)
        else:
            data = pd.read_csv(stint_data_path)
        
        # Basic data cleaning
        data = self._clean_stint_data(data)
        
        # Filter short stints
        min_minutes = 1.0  # Minimum stint length
        data = data[data['minutes_played'] >= min_minutes].copy()
        
        # Calculate target variable (efficiency per 90 minutes)
        data['target'] = (data['total_xG'] / data['minutes_played']) * 90
        
        # Clip extreme values
        data['target'] = data['target'].clip(lower=0, upper=10)
        
        logger.info(f"Prepared {len(data)} stints for modeling")
        return data
    
    def _clean_stint_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate stint data."""
        # Remove rows with missing critical data
        data = data.dropna(subset=['total_xG', 'minutes_played', 'game_id'])
        
        # Ensure numeric types
        numeric_cols = ['total_xG', 'minutes_played']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove invalid values
        data = data[data['minutes_played'] > 0]
        data = data[data['total_xG'] >= 0]
        
        return data.reset_index(drop=True)
    
    def create_feature_matrix(self, data: pd.DataFrame) -> Tuple[csr_matrix, np.ndarray, np.ndarray, List[str]]:
        """
        Create feature matrix for RAPM modeling.
        
        Args:
            data: Prepared stint data
            
        Returns:
            Tuple of (X, y, weights, feature_names)
        """
        logger.info("Creating RAPM feature matrix...")
        
        # Identify player feature columns
        offense_cols = [col for col in data.columns if col.endswith('_offense')]
        defense_cols = [col for col in data.columns if col.endswith('_defense')]
        
        # Filter players by minimum minutes (simplified approach)
        valid_features = self._filter_players_by_minutes(
            offense_cols + defense_cols, data
        )
        
        # Create feature matrix
        X = data[valid_features].values
        y = data['target'].values
        
        # Create weights based on stint length
        weights = data['minutes_played'].values / data['minutes_played'].mean()
        
        # Convert to sparse matrix for efficiency
        X_sparse = csr_matrix(X)
        
        logger.info(f"Feature matrix shape: {X_sparse.shape}")
        logger.info(f"Features: {len(offense_cols)} offense, {len(defense_cols)} defense")
        
        return X_sparse, y, weights, valid_features
    
    def _filter_players_by_minutes(self, feature_cols: List[str], data: pd.DataFrame) -> List[str]:
        """Filter players based on minimum minutes threshold (simplified)."""
        # In production, this would use external minutes data
        # For demo, we approximate based on feature frequency
        
        valid_features = []
        min_appearances = 10  # Minimum stint appearances
        
        for col in feature_cols:
            if data[col].sum() >= min_appearances:
                valid_features.append(col)
        
        logger.info(f"Filtered to {len(valid_features)} valid player features")
        return valid_features
    
    def find_optimal_alpha(self, X: csr_matrix, y: np.ndarray, weights: np.ndarray, 
                          groups: np.ndarray) -> float:
        """
        Find optimal regularization parameter using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            weights: Sample weights
            groups: Group labels for cross-validation
            
        Returns:
            Optimal alpha value
        """
        logger.info("Finding optimal regularization parameter...")
        
        best_alpha = None
        best_score = float('inf')
        
        # Group-based cross-validation (respects game structure)
        cv = GroupKFold(n_splits=self.config['cv_folds'])
        
        for alpha in self.config['alpha_range']:
            model = Ridge(alpha=alpha, solver='auto', max_iter=1000)
            
            # Cross-validation scores
            scores = []
            for train_idx, val_idx in cv.split(X, y, groups):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                w_train = weights[train_idx]
                
                model.fit(X_train, y_train, sample_weight=w_train)
                y_pred = model.predict(X_val)
                
                mse = mean_squared_error(y_val, y_pred)
                scores.append(mse)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
        
        logger.info(f"Optimal alpha: {best_alpha:.2e}, CV RMSE: {np.sqrt(best_score):.4f}")
        return best_alpha
    
    def fit(self, X: csr_matrix, y: np.ndarray, weights: np.ndarray, 
            feature_names: List[str], groups: np.ndarray = None) -> 'RAPMDemo':
        """
        Fit RAPM model to data.
        
        Args:
            X: Feature matrix
            y: Target variable  
            weights: Sample weights
            feature_names: List of feature names
            groups: Group labels for cross-validation
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting RAPM model...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Find optimal regularization if not provided
        if groups is not None:
            optimal_alpha = self.find_optimal_alpha(X, y, weights, groups)
        else:
            optimal_alpha = 1000  # Default value
        
        # Fit final model
        self.model = Ridge(alpha=optimal_alpha, solver='auto', max_iter=1000)
        self.model.fit(X, y, sample_weight=weights)
        
        # Calculate model performance
        y_pred = self.model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_r2 = r2_score(y, y_pred)
        
        logger.info(f"Model fitted - RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        
        return self
    
    def extract_player_ratings(self, min_minutes_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract player ratings from fitted model.
        
        Args:
            min_minutes_data: External data with player minutes (optional)
            
        Returns:
            DataFrame with player ratings
        """
        if self.model is None:
            raise ValueError("Model must be fitted before extracting ratings")
        
        logger.info("Extracting player ratings...")
        
        coefficients = self.model.coef_
        results = []
        
        for i, feature in enumerate(self.feature_names):
            coef = coefficients[i]
            
            # Parse feature name
            if '_offense' in feature:
                player = feature.replace('_offense', '')
                role = 'offense'
            elif '_defense' in feature:
                player = feature.replace('_defense', '')
                role = 'defense'
            else:
                continue
            
            results.append({
                'player': player,
                'role': role,
                'coefficient': coef,
                'impact_per_90': coef  # In demo, no additional adjustments
            })
        
        # Convert to DataFrame and pivot
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("No valid player features found")
            return pd.DataFrame()
        
        # Pivot to get offense/defense columns
        player_ratings = results_df.pivot_table(
            index='player',
            columns='role', 
            values='impact_per_90',
            fill_value=0
        ).reset_index()
        
        # Calculate net impact
        if 'offense' in player_ratings.columns and 'defense' in player_ratings.columns:
            player_ratings['net_impact'] = player_ratings['offense'] - player_ratings['defense']
        else:
            player_ratings['net_impact'] = 0
        
        # Add minutes if available
        if min_minutes_data is not None:
            player_ratings = player_ratings.merge(
                min_minutes_data[['Player', 'Min_Playing']].rename(columns={'Player': 'player'}),
                on='player',
                how='left'
            )
            player_ratings['Min_Playing'] = player_ratings['Min_Playing'].fillna(0)
        
        # Sort by net impact
        player_ratings = player_ratings.sort_values('net_impact', ascending=False)
        
        self.results = player_ratings
        logger.info(f"Extracted ratings for {len(player_ratings)} players")
        
        return player_ratings
    
    def bootstrap_confidence_intervals(self, X: csr_matrix, y: np.ndarray, 
                                     weights: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
        """
        Calculate bootstrap confidence intervals (simplified version).
        
        Note: Production version uses more sophisticated bootstrap procedures.
        
        Args:
            X: Feature matrix
            y: Target variable
            weights: Sample weights
            groups: Group labels
            
        Returns:
            DataFrame with confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be fitted before bootstrap analysis")
        
        logger.info("Running simplified bootstrap analysis...")
        
        n_iterations = self.config['bootstrap_iterations']
        bootstrap_results = []
        
        unique_groups = np.unique(groups)
        
        for i in range(n_iterations):
            # Sample groups with replacement
            np.random.seed(self.config['random_state'] + i)
            sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
            
            # Get indices for sampled groups
            sample_indices = []
            for group in sampled_groups:
                group_indices = np.where(groups == group)[0]
                sample_indices.extend(group_indices)
            
            sample_indices = np.array(sample_indices)
            
            # Fit model on bootstrap sample
            X_boot = X[sample_indices]
            y_boot = y[sample_indices]
            w_boot = weights[sample_indices]
            
            boot_model = Ridge(alpha=self.model.alpha, solver='auto', max_iter=1000)
            boot_model.fit(X_boot, y_boot, sample_weight=w_boot)
            
            # Store coefficients
            bootstrap_results.append(boot_model.coef_)
        
        # Calculate confidence intervals
        bootstrap_array = np.array(bootstrap_results)
        alpha_level = 1 - self.config['confidence_level']
        
        ci_results = []
        for i, feature in enumerate(self.feature_names):
            if '_offense' in feature or '_defense' in feature:
                player = feature.replace('_offense', '').replace('_defense', '')
                role = 'offense' if '_offense' in feature else 'defense'
                
                coef_samples = bootstrap_array[:, i]
                ci_lower = np.percentile(coef_samples, 100 * alpha_level / 2)
                ci_upper = np.percentile(coef_samples, 100 * (1 - alpha_level / 2))
                
                ci_results.append({
                    'player': player,
                    'role': role,
                    'mean_impact': np.mean(coef_samples),
                    'std_impact': np.std(coef_samples),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': not (ci_lower <= 0 <= ci_upper)
                })
        
        return pd.DataFrame(ci_results)
    
    def evaluate_model(self, X: csr_matrix, y: np.ndarray, weights: np.ndarray, 
                      groups: np.ndarray) -> Dict:
        """
        Evaluate model performance using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            weights: Sample weights  
            groups: Group labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info("Evaluating model performance...")
        
        cv = GroupKFold(n_splits=self.config['cv_folds'])
        
        rmse_scores = []
        r2_scores = []
        
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = weights[train_idx]
            
            # Fit model on training fold
            fold_model = Ridge(alpha=self.model.alpha, solver='auto', max_iter=1000)
            fold_model.fit(X_train, y_train, sample_weight=w_train)
            
            # Evaluate on validation fold
            y_pred = fold_model.predict(X_val)
            
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2_scores.append(r2_score(y_val, y_pred))
        
        evaluation = {
            'cv_rmse_mean': np.mean(rmse_scores),
            'cv_rmse_std': np.std(rmse_scores),
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'alpha': self.model.alpha
        }
        
        logger.info(f"CV RMSE: {evaluation['cv_rmse_mean']:.4f} ± {evaluation['cv_rmse_std']:.4f}")
        logger.info(f"CV R²: {evaluation['cv_r2_mean']:.4f} ± {evaluation['cv_r2_std']:.4f}")
        
        return evaluation
    
    def plot_top_players(self, n_players: int = 15, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot top players by net impact.
        
        Args:
            n_players: Number of top players to show
            figsize: Figure size
        """
        if self.results is None:
            raise ValueError("Must extract player ratings before plotting")
        
        plt.figure(figsize=figsize)
        
        # Get top players
        top_players = self.results.head(n_players).copy()
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_players))
        
        plt.barh(y_pos, top_players['net_impact'], alpha=0.8, color='steelblue')
        plt.yticks(y_pos, top_players['player'])
        plt.xlabel('Net RAPM Impact (xG per 90)')
        plt.title(f'Top {n_players} Players by RAPM Net Impact', fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add zero line
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_offense_vs_defense(self, min_minutes: int = 1000, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot offensive vs defensive impact for qualified players.
        
        Args:
            min_minutes: Minimum minutes for inclusion
            figsize: Figure size
        """
        if self.results is None:
            raise ValueError("Must extract player ratings before plotting")
        
        # Filter by minutes if available
        plot_data = self.results.copy()
        if 'Min_Playing' in plot_data.columns:
            plot_data = plot_data[plot_data['Min_Playing'] >= min_minutes]
        
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(plot_data['offense'], plot_data['defense'], 
                   alpha=0.6, s=60, color='steelblue')
        
        # Add reference lines
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Labels and title
        plt.xlabel('Offensive RAPM Impact')
        plt.ylabel('Defensive RAPM Impact')
        plt.title('Player RAPM: Offensive vs Defensive Impact', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add quadrant labels
        plt.text(0.02, 0.98, 'Poor Offense\nGood Defense', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(0.98, 0.98, 'Good Offense\nGood Defense', transform=plt.gca().transAxes,
                horizontalalignment='right', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_dir: str) -> None:
        """
        Save model results to files.
        
        Args:
            output_dir: Directory to save results
        """
        if self.results is None:
            raise ValueError("Must extract player ratings before saving")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save player ratings
        self.results.to_csv(output_path / 'rapm_player_ratings.csv', index=False)
        
        logger.info(f"Results saved to {output_dir}")


def run_rapm_demo(stint_data_path: str, output_dir: str = 'results/rapm_demo',
                  player_minutes_path: Optional[str] = None) -> RAPMDemo:
    """
    Run complete RAPM demo analysis.
    
    Args:
        stint_data_path: Path to processed stint data
        output_dir: Directory for output files
        player_minutes_path: Path to player minutes data (optional)
        
    Returns:
        Fitted RAPM model
    """
    logger.info("Starting RAPM Demo Analysis")
    print("🏈 Soccer RAPM Demo Analysis")
    print("=" * 50)
    
    # Initialize model
    rapm = RAPMDemo()
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    data = rapm.load_and_prepare_data(stint_data_path)
    
    # Create feature matrix
    print("\n🔧 Creating feature matrix...")
    X, y, weights, feature_names = rapm.create_feature_matrix(data)
    groups = data['game_id'].values
    
    # Fit model
    print("\n🎯 Fitting RAPM model...")
    rapm.fit(X, y, weights, feature_names, groups)
    
    # Extract player ratings
    print("\n📈 Extracting player ratings...")
    player_minutes = None
    if player_minutes_path:
        try:
            player_minutes = pd.read_csv(player_minutes_path)
        except FileNotFoundError:
            logger.warning(f"Player minutes file not found: {player_minutes_path}")
    
    ratings = rapm.extract_player_ratings(player_minutes)
    
    # Evaluate model
    print("\n📊 Evaluating model performance...")
    evaluation = rapm.evaluate_model(X, y, weights, groups)
    
    # Display results
    print(f"\n🏆 RAPM Demo Results")
    print("=" * 50)
    print(f"Model Performance:")
    print(f"  • Cross-validation RMSE: {evaluation['cv_rmse_mean']:.4f} ± {evaluation['cv_rmse_std']:.4f}")
    print(f"  • Cross-validation R²: {evaluation['cv_r2_mean']:.4f} ± {evaluation['cv_r2_std']:.4f}")
    print(f"  • Features: {evaluation['n_features']}")
    print(f"  • Samples: {evaluation['n_samples']}")
    
    if len(ratings) > 0:
        print(f"\nTop 10 Players by Net Impact:")
        print("-" * 50)
        top_10 = ratings.head(10)
        for _, row in top_10.iterrows():
            print(f"{row['player']:<25} {row['net_impact']:>8.4f}")
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    try:
        if len(ratings) > 0:
            rapm.plot_top_players()
            if 'offense' in ratings.columns and 'defense' in ratings.columns:
                rapm.plot_offense_vs_defense()
    except Exception as e:
        logger.warning(f"Visualization error: {e}")
    
    # Save results
    print(f"\n💾 Saving results to {output_dir}...")
    rapm.save_results(output_dir)
    
    print(f"\n✅ RAPM Demo Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    
    return rapm


if __name__ == "__main__":
    # Example usage
    stint_data_path = "../data/processed/rapm/rapm_input_clean.pkl"
    player_minutes_path = "../data/raw/player_minutes.csv"
    output_dir = "results/rapm_demo"
    
    try:
        rapm_model = run_rapm_demo(
            stint_data_path=stint_data_path,
            output_dir=output_dir,
            player_minutes_path=player_minutes_path
        )
        
        print(f"\nTo use this model:")
        print(f"1. rapm_model.results contains player ratings")
        print(f"2. Use rapm_model.plot_top_players() for visualizations")
        print(f"3. Results are saved in {output_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data file not found - {e}")
        print(f"Make sure to run the preprocessing pipeline first.")
    except Exception as e:
        print(f"❌ Error running RAPM demo: {e}")
        logger.error(f"RAPM demo failed: {e}", exc_info=True)