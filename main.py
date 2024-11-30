from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Float, Integer, String, Table, MetaData
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import joblib
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fastapi.responses import StreamingResponse

# Initialize FastAPI app
app = FastAPI(title="Financial Prediction API with Visualization", version="4.0")

# Database setup
DATABASE_URL = "sqlite:///./financial_data.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define database schema for financial data
financial_table = Table(
    "financial_data",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("Liabilities", Float),
    Column("Assets", Float),
    Column("Revenue", Float),
    Column("Net Income", Float)
)

# Define database schema for saved models
model_table = Table(
    "saved_models",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("model_name", String, unique=True),
    Column("model_binary", String)  # Store serialized model as base64 string
)

# Create all tables
metadata.create_all(bind=engine)

# Paths and utilities
MODEL_PATH = "best_model.pkl"

# Validate dataset function
def validate_data(df: pd.DataFrame):
    required_columns = ["Liabilities", "Assets", "Revenue", "Net Income"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.isnull().any().any():
        raise ValueError("Data contains null values. Please clean the dataset before uploading.")
    return True

# Define input data model
class PredictionRequest(BaseModel):
    Liabilities: float
    Assets: float
    Revenue: float

# Endpoint: Upload dataset
@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        validate_data(df)
        # Insert data into the database
        with engine.begin() as connection:
            df.to_sql("financial_data", con=connection, if_exists="replace", index=False)
        return {"message": "Dataset uploaded and validated successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the file.")

# Endpoint: Train the model
@app.post("/train/")
async def train_model():
    # Retrieve data from the database
    with engine.connect() as connection:
        df = pd.read_sql_table("financial_data", con=connection)

    # Feature engineering
    df["Revenue Growth Rate"] = df["Revenue"].pct_change().fillna(0)
    df["Asset Turnover Ratio"] = df["Revenue"] / df["Assets"]
    df["Liability to Asset Ratio"] = df["Liabilities"] / df["Assets"]
    print('A')
    features = ["Liabilities", "Assets", "Revenue Growth Rate", "Asset Turnover Ratio", "Liability to Asset Ratio"]
    print('b')
    X = df[features]
    print('c')
    X= X.rename(str,axis="columns")
    print('d')
    y = df["Net Income"]
    print(df.head())
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "SVR": SVR()
    }

    best_model = None
    best_score = -np.inf
    best_model_name = ""

    # Train and evaluate models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        print(f"{model_name} R² score: {score}")

        # Save the best model
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = model_name

    # Save the best model to disk and database
    if best_model is not None:
        joblib.dump(best_model, MODEL_PATH)

        return {
            "message": f"Best model '{best_model_name}' trained and saved successfully.",
            "best_model": best_model_name,
            "best_r2_score": best_score
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to train models.")

@app.post("/predict/")
async def predict(request: PredictionRequest):
    model = joblib.load(MODEL_PATH)

    # Prepare input data for prediction
    input_data = pd.DataFrame([request.dict()])

    # Apply feature engineering
    input_data["Revenue Growth Rate"] = input_data["Revenue"].pct_change().fillna(0)
    input_data["Asset Turnover Ratio"] = input_data["Revenue"] / input_data["Assets"]
    input_data["Liability to Asset Ratio"] = input_data["Liabilities"] / input_data["Assets"]

    # Ensure the correct order of features
    features = ["Liabilities", "Assets", "Revenue Growth Rate", "Asset Turnover Ratio", "Liability to Asset Ratio"]
    input_data = input_data[features]

    # Predict with the trained model
    prediction = model.predict(input_data)

    return {"predicted_net_income": prediction[0]}

# Visualization utility function
def generate_plot(df, plot_type="histogram"):
    plt.figure(figsize=(10, 6))
    if plot_type == "histogram":
        df.hist(bins=20, color='skyblue', edgecolor='black')
        plt.suptitle("Dataset Histogram")
    elif plot_type == "scatter":
        if "Revenue" in df.columns and "Net Income" in df.columns:
            plt.scatter(df["Revenue"], df["Net Income"], alpha=0.7, color='orange')
            plt.title("Revenue vs Net Income")
            plt.xlabel("Revenue")
            plt.ylabel("Net Income")
    elif plot_type == "correlation":
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Endpoint: Visualize dataset
@app.get("/visualize/{plot_type}")
async def visualize_data(plot_type: str):
    with engine.connect() as connection:
        df = pd.read_sql_table("financial_data", con=connection)

    try:
        buf = generate_plot(df, plot_type=plot_type)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
