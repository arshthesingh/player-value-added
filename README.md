# Performance-ROI Analytics Framework using PVA

## Executive Summary
This project implements a sophisticated player performance evaluation system combining **Regularized Adjusted Plus-Minus (RAPM)** and **Statistical Plus-Minus (SPM)** methodologies. Using advanced statistical techniques including a multi-stage Bayesian analytics framework, and cross-validation, the system accurately predicts team performance and evaluates individual player contributions.

### Key Achievements:

- 🎯 0.848 correlation with actual league standings across 5 major competitions
- 📊 79% accuracy within 3 positions for team ranking predictions
- 🔄 0.741-0.809 stability in year-over-year player ratings (1000+ minutes)
- ⚡ 2.1 average position difference between predicted and actual final standings

Business Impact: Enables data-driven decision making for efficiency, ROI, strategic planning, and performance optimization with quantifiable accuracy metrics.

## Methodology Overview
### Problem Statement
Traditional performance metrics fail to account for contextual factors such as teammate quality, opposition strength, and situational effects. This system addresses these limitations through advanced statistical modeling techniques.
### Solution Architecture
The system employs a three-stage approach:
1. RAPM (Regularized Adjusted Plus-Minus)
- Event-driven stint analysis isolating individual contributions
- Advanced regularization preventing overfitting
- Cross-game validation ensuring generalizability
2. SPM (Statistical Plus-Minus)
- Bayesian combination of multi-season priors with current performance
- Feature selection using elastic net regularization
- Reliability-based shrinkage for small sample sizes
3. Team Adjustments
- BPM-style team context incorporation
- League-specific performance normalization
- Cross-temporal validation


### Mathematical Framework
### RAPM Model:
y_i = α + Σ(β_j × X_ij) + Σ(γ_k × D_ik) + ε_i

Where:
- y_i = target metric for stint i
- X_ij = indicator for player j on offense
- D_ik = indicator for player k on defense
- β_j, γ_k = player coefficients (regularized)