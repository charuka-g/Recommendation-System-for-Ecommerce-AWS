# Two-Stage Recommendation System

A production-ready two-stage recommendation system built on AWS SageMaker:
1. **Retrieval Stage**: Matrix Factorization (using Surprise library) for candidate generation
2. **Ranking Stage**: XGBoost for final ranking using Feature Store metadata

## Architecture

![Two-Stage Recommendation System Architecture](recommedaion_system_aws.drawio.png)

```
User Request
    ↓
Stage 1: Retrieval (Matrix Factorization)
    ├── Compute user-item similarity using embeddings
    ├── Retrieve top-100 candidates
    └── Exclude already-interacted items
    ↓
Stage 2: Ranking (XGBoost)
    ├── Fetch metadata from Feature Store
    ├── Enrich candidates with features
    ├── Rank using XGBoost model
    └── Return top-K recommendations
```

## Project Structure

```
Recommendation-System/
├── .env                              # Environment variables (gitignored)
├── .env.example                      # Example environment file
├── 00-Setup-Configuration.ipynb      # Setup and AWS/SageMaker configuration
├── 01-Train-FM-Model.ipynb          # Train Matrix Factorization model
├── 02-Train-XGBoost-Model.ipynb     # Train XGBoost ranking model
├── 03-Glue-Feature-Store-Processing.ipynb  # AWS Glue job for Feature Store data
├── 03-Inference-Pipeline.ipynb      # Development/testing inference (optional)
├── inference_api.py                  # Production FastAPI endpoint
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Getting Started

### Prerequisites

- AWS Account with SageMaker access
- IAM role with SageMaker, S3, Feature Store, and DynamoDB permissions
- Python 3.8+
- Feature Groups created (User and Item Feature Groups)

### Installation

```bash
pip install -r requirements.txt
```

### Step 0: Configure Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# AWS Configuration
AWS_REGION=ap-south-1
AWS_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE_NAME
S3_BUCKET=your-s3-bucket-name

# Feature Store Configuration
USER_FEATURE_GROUP_NAME=all-beauty-users-<timestamp>
ITEM_FEATURE_GROUP_NAME=all-beauty-items-<timestamp>

# DynamoDB Configuration
DYNAMODB_TABLE_NAME=user-candidates

# SageMaker Endpoints (set after training)
XGBOOST_ENDPOINT_NAME=

# Feature Names (JSON array, set after XGBoost training)
FEATURE_NAMES=[]

# Model Paths
MF_MODEL_PATH=/opt/ml/model/mf_model_artifacts.pkl
```

**Note**: The `.env` file is gitignored for security. Never commit sensitive credentials.

### Step 1: Setup Configuration

Run `00-Setup-Configuration.ipynb` to:
- Load configuration from `.env` file
- Initialize AWS/SageMaker sessions
- Validate configuration
- List available Feature Groups

### Step 2: Train Matrix Factorization Model

Run `01-Train-FM-Model.ipynb` to:
- Load interaction data (user_id, parent_asin, rating)
- Prepare training data for Surprise library
- Train Matrix Factorization model using Surprise SVD
- Extract user and item embeddings
- Build K-NN index for candidate retrieval
- Save model artifacts to S3
- Register model in SageMaker Model Registry
- Generate and store candidates in DynamoDB

**Key Functions** (defined in notebook):
- `prepare_mf_training_data()`: Prepares data in Surprise format
- `train_matrix_factorization()`: Trains SVD model and extracts embeddings
- `build_knn_index()`: Builds K-NN index for item embeddings
- `retrieve_top_k_candidates()`: Retrieves top-K candidates using embeddings
- `save_mf_model_to_registry()`: Registers model in Model Registry
- `store_candidates_in_dynamodb()`: Pre-computes candidates for all users

### Step 3: Train XGBoost Ranking Model

Run `02-Train-XGBoost-Model.ipynb` to:
- Load Feature Groups (User and Item)
- Prepare ranking training data with point-in-time Feature Store joins
- Train XGBoost model for rating prediction
- Deploy XGBoost endpoint
- Save model information (endpoint name, feature names)

**Key Functions** (defined in notebook):
- `prepare_ranking_training_data()`: Creates dataset with Feature Store joins
- `train_xgboost_ranker()`: Trains XGBoost model on SageMaker

### Step 4: Feature Store Data Processing (Optional)

Run `03-Glue-Feature-Store-Processing.ipynb` if you need to process raw data for Feature Store:
- Processes reviews and metadata datasets
- Creates User and Item Feature Groups
- Handles event time for point-in-time accuracy

### Step 5: Testing (Optional)

Run `03-Inference-Pipeline.ipynb` for development/testing:
- Loads trained models from artifacts
- Tests the complete recommendation pipeline
- Useful for debugging and validation

**Note**: This notebook is optional. For production, use `inference_api.py`.

## Production Deployment

### FastAPI Inference Endpoint

The `inference_api.py` file provides a production-ready FastAPI endpoint for serving recommendations.

#### Environment Variables

Set the following environment variables:

```bash
AWS_REGION=ap-south-1
DYNAMODB_TABLE_NAME=user-candidates
XGBOOST_ENDPOINT_NAME=your-xgboost-endpoint-name
USER_FEATURE_GROUP_NAME=all-beauty-users-<timestamp>
ITEM_FEATURE_GROUP_NAME=all-beauty-items-<timestamp>
FEATURE_NAMES=["item_price", "item_category", ...]  # JSON array
MF_MODEL_PATH=/opt/ml/model/mf_model_artifacts.pkl  # Optional, for fallback
```

#### Running the API

```bash
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

- `GET /` - Health check
- `GET /health` - Health check
- `POST /recommendations` - Get recommendations

**Request**:
```json
{
  "user_id": "USER_ID_HERE",
  "top_k": 10
}
```

**Response**:
```json
{
  "user_id": "USER_ID_HERE",
  "recommendations": [
    {
      "parent_asin": "ITEM_ID",
      "predicted_rating": 4.5
    }
  ],
  "num_recommendations": 10
}
```

## Workflow Details

### Stage 1: Retrieval (Matrix Factorization)

1. **Training**:
   - Uses Surprise library's SVD algorithm
   - Learns user and item embeddings (64-dimensional vectors)
   - Trained on user-item interaction data

2. **Inference**:
   - Computes user-item similarity using dot product
   - Retrieves top-100 candidate items
   - Excludes items user has already interacted with

3. **Optimization**:
   - Candidates are pre-computed and stored in DynamoDB
   - Production API reads from DynamoDB (faster)
   - Development notebook computes on-the-fly (for testing)

### Stage 2: Ranking (XGBoost)

1. **Feature Enrichment**:
   - Fetches item features: Price, Category, Average Rating, Rating Count
   - Fetches user features: User Rating Count
   - Uses point-in-time accurate Feature Store queries

2. **Ranking**:
   - XGBoost model predicts rating for each candidate
   - Features are one-hot encoded to match training format
   - Returns top-K items sorted by predicted rating

## Configuration

All configuration is managed through the `.env` file. The notebooks automatically load configuration from this file using `python-dotenv`.

### Required Configuration

- `AWS_REGION`: AWS region (e.g., `ap-south-1`)
- `AWS_ROLE_ARN`: IAM role ARN for SageMaker
- `S3_BUCKET`: S3 bucket name for storing artifacts
- `USER_FEATURE_GROUP_NAME`: Name of User Feature Group
- `ITEM_FEATURE_GROUP_NAME`: Name of Item Feature Group
- `DYNAMODB_TABLE_NAME`: DynamoDB table name for candidates (default: `user-candidates`)

### Optional Configuration

- `XGBOOST_ENDPOINT_NAME`: Set after deploying XGBoost model
- `FEATURE_NAMES`: JSON array of feature names (set after XGBoost training)
- `MF_MODEL_PATH`: Path to MF model artifacts (for production deployment)

### Updating Configuration

Simply edit the `.env` file and restart your notebook or API server. No code changes needed!

## Requirements

See `requirements.txt` for full list. Key dependencies:

- `sagemaker` - AWS SageMaker SDK
- `boto3` - AWS SDK
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `surprise` (scikit-surprise) - Matrix Factorization
- `fastapi` - API framework (for inference_api.py)
- `uvicorn` - ASGI server (for inference_api.py)

## Key Features

- **Point-in-time accurate joins**: Feature Store ensures historical accuracy
- **Scalable**: Pre-computed candidates in DynamoDB for fast retrieval
- **Production-ready**: FastAPI endpoint for deployment
- **Modular**: Separate notebooks for each stage
- **Self-contained**: All functions defined in notebooks (no external modules)
- **Model Registry**: Models registered in SageMaker Model Registry
- **Feature Store integration**: Real-time feature retrieval

## Notes

- Feature Groups must be created before training XGBoost
- Matrix Factorization uses Surprise library's SVD (optimized for collaborative filtering)
- Model artifacts are saved to S3
- Candidates are pre-computed and stored in DynamoDB for production
- The `03-Inference-Pipeline.ipynb` notebook is optional (for development/testing)
- Production inference uses `inference_api.py` which reads from DynamoDB

## Troubleshooting

1. **Feature Groups not found**: Update Feature Group names in configuration
2. **DynamoDB table missing**: Create table with `user_id` as partition key
3. **XGBoost endpoint not found**: Ensure endpoint is deployed from `02-Train-XGBoost-Model.ipynb`
4. **Missing features**: Verify Feature Store has data ingested

Copyright © 2026 Charuka Gunawardhane. All rights reserved.


