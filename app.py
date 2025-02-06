from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed asset data
processed_data = pd.read_csv("processed_assets_final.csv")  # Ensure this file exists

# Load LabelEncoder for ticker decoding
le_ticker = joblib.load("label_encoder_ticker.joblib")  # Load saved LabelEncoder

# Define mappings for encoding user inputs
knowledge_mapping = {"principiante": 1, "intermedio": 2, "avanzado": 3}
risk_mapping = {"bajo": 1, "moderado": 2, "alto": 3}

# Initialize FastAPI
app = FastAPI()

# Define request data model
class UserProfile(BaseModel):
    knowledgeLevel: str
    goals: list
    riskPreference: str
    monthlyIncome: int
    monthlyExpenses: int
    savingsPercentage: float

# Recommendation function
def recommend_assets(user_data, processed_data, le_ticker, knowledge_mapping, risk_mapping):
    # Normalize user financial values
    max_income, max_expenses = 1700000, 1200000
    user_profile = {
        "income_normalized": user_data.monthlyIncome / max_income,
        "expenses_normalized": user_data.monthlyExpenses / max_expenses,
        "savings_percentage": user_data.savingsPercentage / 100
    }

    # Encode knowledge level & risk preference
    user_profile["knowledge_level_encoded"] = knowledge_mapping[user_data.knowledgeLevel]
    user_profile["risk_level_encoded"] = risk_mapping[user_data.riskPreference]

    # One-Hot Encode Investment Goals
    investment_goals = ["retiro", "bienes", "proyectos", "vacaciones"]
    for goal in investment_goals:
        user_profile[f"goal_{goal}"] = 1 if goal in user_data.goals else 0

    # Convert to DataFrame
    user_df = pd.DataFrame([user_profile])

    # Compute Cosine Similarity
    asset_features = processed_data.drop(columns=["ticker_encoded"])
    similarity_scores = cosine_similarity(user_df, asset_features)

    # Convert similarity scores to DataFrame
    similarity_df = pd.DataFrame(similarity_scores.flatten(), index=processed_data.index, columns=["similarity"])

    # Merge similarity scores with asset details
    recommendations = processed_data.copy()
    recommendations["similarity"] = similarity_df["similarity"]
    recommendations["ticker"] = le_ticker.inverse_transform(recommendations["ticker_encoded"])

    # Get unique top 3 recommendations
    top_recommendations = recommendations.sort_values(by="similarity", ascending=False).head(2)
    #top_recommendations = top_recommendations.drop_duplicates(subset=["ticker"]).head(3)

    return top_recommendations[["ticker"]].reset_index(drop=True).to_dict(orient="records")

# API route for recommendations
@app.post("/recommend")
def get_recommendation(user: UserProfile):
    recommendations = recommend_assets(user, processed_data, le_ticker, knowledge_mapping, risk_mapping)

    return {"recommended_assets": recommendations}
