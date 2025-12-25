# Two-Stage Recommendation System

A production-ready two-stage recommendation system built on AWS SageMaker:
1. **Retrieval Stage**: Matrix Factorization for candidate generation
2. **Ranking Stage**: XGBoost for final ranking using Feature Store metadata

## Project Structure

```
Recommendation-System/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and session setup
│   ├── mf_retrieval.py          # Matrix Factorization functions
│   ├── xgboost_ranking.py       # XGBoost ranking functions
│   ├── feature_store_utils.py   # Feature Store utilities
│   └── recommender.py           # TwoStageRecommender class
│
├── 00-Setup-Configuration.ipynb  # Setup and configuration
├── 01-Train-FM-Model.ipynb      # Train Matrix Factorization
├── 02-Train-XGBoost-Model.ipynb # Train XGBoost ranking model
├── 03-Inference-Pipeline.ipynb  # Inference and recommendations
│
└── Other notebooks...
```

## Getting Started

### 1. Setup Configuration

Run `00-Setup-Configuration.ipynb` to:
- Initialize AWS/SageMaker sessions
- Configure Feature Group names
- Set up environment variables

### 2. Train Matrix Factorization Model

Run `01-Train-FM-Model.ipynb` to:
- Load interaction data
- Prepare training data for MF
- Train the Matrix Factorization model using PyTorch
- Get user and item embeddings
- Build K-NN index
- Save model artifacts

### 3. Train XGBoost Ranking Model

Run `02-Train-XGBoost-Model.ipynb` to:
- Load Feature Groups
- Prepare ranking training data with Feature Store joins
- Train XGBoost model
- Deploy XGBoost endpoint
- Save model information

### 4. Run Inference

Run `03-Inference-Pipeline.ipynb` to:
- Load trained models
- Initialize TwoStageRecommender
- Get recommendations for users

## Source Code Modules

### `src/config.py`
- `get_config()`: Get configuration dictionary
- `initialize_sessions()`: Initialize AWS and SageMaker sessions

### `src/mf_retrieval.py`
- `prepare_mf_training_data()`: Prepare data for Matrix Factorization
- `train_matrix_factorization()`: Train MF model using PyTorch
- `build_knn_index()`: Build K-NN index for item embeddings
- `retrieve_top_k_candidates()`: Retrieve top-K candidates using embeddings

### `src/xgboost_ranking.py`
- `prepare_ranking_training_data()`: Prepare training data with Feature Store joins
- `train_xgboost_ranker()`: Train XGBoost ranking model

### `src/feature_store_utils.py`
- `fetch_feature_store_metadata()`: Fetch metadata from Online Feature Store

### `src/recommender.py`
- `TwoStageRecommender`: Main recommender class
- `get_recommendations()`: Standalone function for getting recommendations

## Usage Example

```python
from src.recommender import get_recommendations

# Initialize recommender (see 03-Inference-Pipeline.ipynb)
recommendations = get_recommendations(
    user_id="USER_ID_HERE",
    recommender=recommender,
    top_k=10
)

for rec in recommendations:
    print(f"Item: {rec['parent_asin']}, Score: {rec['predicted_rating']:.3f}")
```

## Workflow

1. **Retrieval Stage**: 
   - Uses Matrix Factorization embeddings to find top-100 candidate items
   - Computes user-item similarity scores using dot product
   - Excludes already-interacted items

2. **Ranking Stage**:
   - Fetches rich metadata from Feature Store (Price, Category, Ratings, etc.)
   - Uses XGBoost to predict ratings for candidates
   - Returns top-K items sorted by predicted rating

## Configuration

Update `src/config.py` with your:
- AWS region
- IAM role ARN
- S3 bucket name
- Feature Group names

## Requirements

- Python 3.8+
- sagemaker
- boto3
- pandas
- numpy
- scikit-learn
- surprise (scikit-surprise for Matrix Factorization)

## Notes

- Feature Groups must be created before training XGBoost
- Matrix Factorization model artifacts are saved locally as `mf_model_artifacts.pkl`
- XGBoost model info is saved as `xgb_model_info.json`
- Point-in-time accurate feature joins are handled automatically
- Matrix Factorization uses Surprise library's SVD algorithm (optimized for collaborative filtering)

