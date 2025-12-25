"""
FastAPI application for online inference in ECS.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import boto3
import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Two-Stage Recommendation System API")

# Initialize AWS clients
region = os.getenv("AWS_REGION", "ap-south-1")
boto_session = boto3.Session(region_name=region)
dynamodb = boto_session.resource('dynamodb')
sagemaker_runtime = boto_session.client('sagemaker-runtime')
featurestore_runtime = boto_session.client('sagemaker-featurestore-runtime')

# Configuration
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "user-candidates")
XGBOOST_ENDPOINT_NAME = os.getenv("XGBOOST_ENDPOINT_NAME", "")
USER_FEATURE_GROUP_NAME = os.getenv("USER_FEATURE_GROUP_NAME", "")
ITEM_FEATURE_GROUP_NAME = os.getenv("ITEM_FEATURE_GROUP_NAME", "")

# Load MF model artifacts (embeddings) - load once at startup
# In production, load from S3
MF_MODEL_PATH = os.getenv("MF_MODEL_PATH", "/opt/ml/model/mf_model_artifacts.pkl")
mf_model = None

try:
    with open(MF_MODEL_PATH, 'rb') as f:
        mf_model = pickle.load(f)
    print("MF model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load MF model: {e}")


class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 10


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, float]]
    num_recommendations: int


def get_candidates_from_dynamodb(user_id: str, k: int = 100) -> List[str]:
    """
    Get pre-computed candidates from DynamoDB.
    
    Args:
        user_id: User ID
        k: Number of candidates to retrieve
    
    Returns:
        List of candidate item IDs
    """
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    
    try:
        response = table.get_item(Key={'user_id': user_id})
        if 'Item' in response:
            candidates = response['Item'].get('candidates', [])
            return candidates[:k]
        else:
            return []
    except Exception as e:
        print(f"Error fetching from DynamoDB: {e}")
        return []


def fetch_feature_store_metadata(
    candidate_items: List[str],
    user_id: str
) -> pd.DataFrame:
    """
    Fetch metadata from Feature Store for candidates.
    
    Args:
        candidate_items: List of item IDs
        user_id: User ID
    
    Returns:
        DataFrame with enriched features
    """
    enriched_features = []
    
    # Fetch user features
    user_features = None
    try:
        user_record = featurestore_runtime.get_record(
            FeatureGroupName=USER_FEATURE_GROUP_NAME,
            RecordIdentifierValueAsString=user_id
        )
        if 'Record' in user_record:
            user_features = {
                feat['FeatureName']: feat['ValueAsString']
                for feat in user_record['Record']
            }
    except Exception as e:
        print(f"Warning: Could not fetch user features: {e}")
    
    # Fetch item features
    for item_id in candidate_items:
        try:
            item_record = featurestore_runtime.get_record(
                FeatureGroupName=ITEM_FEATURE_GROUP_NAME,
                RecordIdentifierValueAsString=item_id
            )
            
            if 'Record' in item_record:
                item_features = {
                    feat['FeatureName']: feat['ValueAsString']
                    for feat in item_record['Record']
                }
                
                combined = {
                    'parent_asin': item_id,
                    'user_id': user_id
                }
                
                # Extract features
                combined['item_price'] = float(item_features.get('price', 0))
                combined['item_category'] = item_features.get('main_category', '')
                combined['item_avg_rating'] = float(item_features.get('average_rating', 0))
                combined['item_rating_count'] = float(item_features.get('rating_count', 0))
                
                if user_features:
                    combined['user_rating_count'] = float(
                        user_features.get('rating_count_by_user', 0)
                    )
                else:
                    combined['user_rating_count'] = 0.0
                
                enriched_features.append(combined)
        except Exception as e:
            print(f"Warning: Could not fetch features for {item_id}: {e}")
            continue
    
    if not enriched_features:
        return pd.DataFrame()
    
    return pd.DataFrame(enriched_features)


def rank_with_xgboost(features_df: pd.DataFrame, feature_names: List[str]) -> List[float]:
    """
    Rank candidates using XGBoost endpoint.
    
    Args:
        features_df: DataFrame with features
        feature_names: List of feature names in order
    
    Returns:
        List of predicted ratings
    """
    # Prepare features (match training format)
    feature_data = features_df.copy()
    
    # Handle categorical features
    categorical_cols = feature_data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['user_id', 'parent_asin']]
    
    if len(categorical_cols) > 0:
        feature_data = pd.get_dummies(feature_data, columns=categorical_cols, drop_first=True)
    
    # Ensure all features are present
    for feat in feature_names:
        if feat not in feature_data.columns:
            feature_data[feat] = 0
    
    # Select features in correct order
    X = feature_data[feature_names]
    X = X.fillna(0)
    
    # Convert to CSV
    csv_data = X.to_csv(index=False, header=False)
    
    # Get predictions
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=XGBOOST_ENDPOINT_NAME,
        ContentType='text/csv',
        Body=csv_data.encode('utf-8')
    )
    
    predictions = response['Body'].read().decode('utf-8')
    pred_scores = [float(x) for x in predictions.strip().split('\n') if x.strip()]
    
    return pred_scores


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "two-stage-recommendation-system"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """
    Get recommendations for a user using the two-stage system.
    
    Stage 1: Retrieve candidates from DynamoDB (pre-computed)
    Stage 2: Rank candidates using XGBoost with Feature Store metadata
    """
    user_id = request.user_id
    top_k = request.top_k
    
    # Stage 1: Get candidates from DynamoDB
    candidate_items = get_candidates_from_dynamodb(user_id, k=100)
    
    if len(candidate_items) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No candidates found for user {user_id}"
        )
    
    # Stage 2: Fetch metadata from Feature Store
    features_df = fetch_feature_store_metadata(candidate_items, user_id)
    
    if len(features_df) == 0:
        raise HTTPException(
            status_code=500,
            detail="Could not fetch features from Feature Store"
        )
    
    # Stage 3: Rank with XGBoost
    # Load feature names (in production, load from S3 or environment)
    feature_names = json.loads(os.getenv("FEATURE_NAMES", "[]"))
    
    if not feature_names:
        # Fallback: extract from DataFrame
        feature_names = [col for col in features_df.columns 
                        if col not in ['user_id', 'parent_asin']]
    
    pred_scores = rank_with_xgboost(features_df, feature_names)
    
    # Combine results
    results = [
        {
            'parent_asin': item,
            'predicted_rating': score
        }
        for item, score in zip(candidate_items[:len(pred_scores)], pred_scores)
    ]
    
    # Sort by predicted rating
    results.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    # Return top-K
    return RecommendationResponse(
        user_id=user_id,
        recommendations=results[:top_k],
        num_recommendations=len(results[:top_k])
    )


# Note: Run with: uvicorn inference_api:app --host 0.0.0.0 --port 8000

