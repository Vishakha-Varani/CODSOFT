# =========================================================
# MOVIE GENRE CLASSIFICATION PROJECT
# =========================================================

# -------------------------
# IMPORT LIBRARIES
# -------------------------

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# =========================================================
# LOAD DATASET
# =========================================================

print("Loading dataset...")

data = pd.read_csv(
    r"C:\Users\User\Downloads\archive (17)\Genre Classification Dataset\train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

print("Dataset Loaded Successfully!")

# =========================================================
# SHOW DATA
# =========================================================

print("\nFirst 5 Rows:\n")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

# =========================================================
# FEATURES AND LABELS
# =========================================================

X = data["DESCRIPTION"]
y = data["GENRE"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# CREATE MACHINE LEARNING PIPELINE
# =========================================================

print("\nCreating model...")

model = Pipeline([
    
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
    ),

    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )

])

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# =========================================================
# MAKE PREDICTIONS
# =========================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

# =========================================================
# EVALUATE MODEL
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

print(classification_report(y_test, y_pred))

# =========================================================
# MOVIE GENRE PREDICTION SYSTEM
# =========================================================

print("\n===================================")
print("MOVIE GENRE PREDICTION SYSTEM")
print("===================================")

while True:

    print("\nEnter Movie Plot Summary:\n")

    user_input = input(">>> ")

    prediction = model.predict([user_input])

    print("\nPredicted Genre:", prediction[0])

    choice = input("\nPredict Again? (y/n): ")

    if choice.lower() != 'y':
        break

print("\nProject Finished Successfully!")