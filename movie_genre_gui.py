# =========================================================
# MOVIE GENRE CLASSIFICATION WITH GUI
# =========================================================

# IMPORT LIBRARIES

import pandas as pd
import warnings
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
# FEATURES AND LABELS
# =========================================================

X = data["DESCRIPTION"]
y = data["GENRE"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# CREATE MODEL
# =========================================================

print("Training model...")

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

model.fit(X_train, y_train)

print("Training completed!")

# =========================================================
# GUI FUNCTION
# =========================================================

def predict_genre():

    movie_plot = text_box.get("1.0", tk.END).strip()

    if movie_plot == "":
        messagebox.showwarning("Warning", "Please enter a movie plot.")
        return

    prediction = model.predict([movie_plot])

    result_label.config(
        text=f"Predicted Genre: {prediction[0]}",
        fg="blue"
    )

# =========================================================
# CREATE GUI WINDOW
# =========================================================

root = tk.Tk()

root.title("Movie Genre Classification")
root.geometry("700x500")
root.config(bg="white")

# =========================================================
# TITLE
# =========================================================

title = tk.Label(
    root,
    text="Movie Genre Classification",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="darkgreen"
)

title.pack(pady=20)

# =========================================================
# INSTRUCTION LABEL
# =========================================================

instruction = tk.Label(
    root,
    text="Enter Movie Plot Summary:",
    font=("Arial", 14),
    bg="white"
)

instruction.pack()

# =========================================================
# TEXT BOX
# =========================================================

text_box = tk.Text(
    root,
    height=10,
    width=70,
    font=("Arial", 12)
)

text_box.pack(pady=10)

# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = tk.Button(
    root,
    text="Predict Genre",
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    padx=20,
    pady=10,
    command=predict_genre
)

predict_button.pack(pady=20)

# =========================================================
# RESULT LABEL
# =========================================================

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="white"
)

result_label.pack(pady=20)

# =========================================================
# RUN GUI
# =========================================================

root.mainloop()