import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- GLOBAL VARIABLES ---------------- #

train_data = None
test_data = None

models = {}
accuracies = {}

scaler = StandardScaler()

feature_columns = []
target_column = None

# ---------------- LOAD TRAIN DATA ---------------- #

def load_train_data():

    global train_data

    file_path = filedialog.askopenfilename(
        title="Select Train CSV",
        filetypes=[("CSV Files", "*.csv")]
    )

    if file_path:

        train_data = pd.read_csv(file_path)

        train_data.columns = train_data.columns.str.strip()

        print("\nTrain Columns:")
        print(train_data.columns)

        messagebox.showinfo(
            "Success",
            "Train CSV Loaded Successfully!"
        )

# ---------------- LOAD TEST DATA ---------------- #

def load_test_data():

    global test_data

    file_path = filedialog.askopenfilename(
        title="Select Test CSV",
        filetypes=[("CSV Files", "*.csv")]
    )

    if file_path:

        test_data = pd.read_csv(file_path)

        test_data.columns = test_data.columns.str.strip()

        print("\nTest Columns:")
        print(test_data.columns)

        messagebox.showinfo(
            "Success",
            "Test CSV Loaded Successfully!"
        )

# ---------------- FIND TARGET COLUMN ---------------- #

def get_target_column(data):

    possible_targets = [

        "Class",
        "class",
        "target",
        "Target",
        "label",
        "is_fraud",
        "fraud"
    ]

    for col in possible_targets:

        if col in data.columns:

            return col

    return None

# ---------------- TRAIN MODELS ---------------- #

def train_models():

    global train_data, test_data
    global models, accuracies
    global scaler
    global feature_columns
    global target_column

    if train_data is None or test_data is None:

        messagebox.showerror(
            "Error",
            "Please load train and test CSV files first!"
        )

        return

    try:

        # Find target column
        target_column = get_target_column(train_data)

        if target_column is None:

            messagebox.showerror(
                "Error",
                "Target column not found!"
            )

            return

        # ---------------- SMALL SAMPLE FOR FAST TRAINING ---------------- #

        train_sample = train_data.sample(
            n=min(5000, len(train_data)),
            random_state=42
        )

        test_sample = test_data.sample(
            n=min(1000, len(test_data)),
            random_state=42
        )

        # ---------------- FEATURES & LABELS ---------------- #

        X_train = train_sample.drop(target_column, axis=1)

        y_train = train_sample[target_column]

        X_test = test_sample.drop(target_column, axis=1)

        y_test = test_sample[target_column]

        # ---------------- CONVERT TEXT TO NUMBERS ---------------- #

        X_train = pd.get_dummies(X_train)

        X_test = pd.get_dummies(X_test)

        # ---------------- MATCH COLUMNS ---------------- #

        X_train, X_test = X_train.align(
            X_test,
            join='left',
            axis=1,
            fill_value=0
        )

        # Save columns
        feature_columns = X_train.columns.tolist()

        # ---------------- SCALE DATA ---------------- #

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        # ---------------- MODELS ---------------- #

        model_dict = {

            "Logistic Regression":
            LogisticRegression(max_iter=500),

            "Decision Tree":
            DecisionTreeClassifier(max_depth=10),

            "Random Forest":
            RandomForestClassifier(
                n_estimators=20,
                max_depth=10
            )
        }

        result_box.delete(1.0, tk.END)

        accuracies.clear()

        # ---------------- TRAIN ---------------- #

        for name, model in model_dict.items():

            result_box.insert(
                tk.END,
                f"Training {name}...\n"
            )

            root.update()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            models[name] = model

            accuracies[name] = accuracy

            result_box.insert(
                tk.END,
                f"{name} Accuracy: {accuracy:.4f}\n\n"
            )

        messagebox.showinfo(
            "Success",
            "Models Trained Successfully!"
        )

    except Exception as e:

        messagebox.showerror(
            "Training Error",
            str(e)
        )

# ---------------- SHOW GRAPH ---------------- #

def show_graph():

    if not accuracies:

        messagebox.showerror(
            "Error",
            "Train models first!"
        )

        return

    names = list(accuracies.keys())

    values = list(accuracies.values())

    plt.figure(figsize=(8,5))

    plt.bar(
        names,
        values,
        color=["blue", "green", "orange"]
    )

    plt.title("Model Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.ylim(0.5, 1.0)

    plt.show()

# ---------------- PREDICT ---------------- #

def predict_transaction():

    if "Random Forest" not in models:

        messagebox.showerror(
            "Error",
            "Train models first!"
        )

        return

    try:

        values = []

        for entry in entries:

            val = entry.get()

            if val == "":
                val = 0

            values.append(float(val))

        input_data = np.array(values).reshape(1, -1)

        # Resize input automatically
        expected = len(feature_columns)

        current = input_data.shape[1]

        if current < expected:

            extra = np.zeros((1, expected - current))

            input_data = np.hstack((input_data, extra))

        elif current > expected:

            input_data = input_data[:, :expected]

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = models["Random Forest"].predict(input_scaled)

        if prediction[0] == 1:

            result = "Fraudulent Transaction"

        else:

            result = "Legitimate Transaction"

        messagebox.showinfo(
            "Prediction Result",
            result
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

# ---------------- GUI ---------------- #

root = tk.Tk()

root.title("Credit Card Fraud Detection")

root.geometry("900x700")

root.configure(bg="#f0f0f0")

# ---------------- TITLE ---------------- #

title = tk.Label(

    root,

    text="Credit Card Fraud Detection System",

    font=("Arial", 22, "bold"),

    bg="#f0f0f0",

    fg="darkblue"
)

title.pack(pady=10)

# ---------------- BUTTONS ---------------- #

button_frame = tk.Frame(root, bg="#f0f0f0")

button_frame.pack(pady=10)

buttons = [

    ("Load Train CSV", load_train_data, "blue"),

    ("Load Test CSV", load_test_data, "purple"),

    ("Train Models", train_models, "green"),

    ("Show Graph", show_graph, "orange")
]

for i, (text, command, color) in enumerate(buttons):

    btn = tk.Button(

        button_frame,

        text=text,

        command=command,

        bg=color,

        fg="white",

        font=("Arial", 12),

        width=15
    )

    btn.grid(row=0, column=i, padx=10)

# ---------------- RESULT BOX ---------------- #

result_box = tk.Text(

    root,

    width=70,

    height=12,

    font=("Arial", 12)
)

result_box.pack(pady=20)

# ---------------- INPUT SECTION ---------------- #

label = tk.Label(

    root,

    text="Enter 10 Numeric Values",

    font=("Arial", 16, "bold"),

    bg="#f0f0f0"
)

label.pack()

input_frame = tk.Frame(root, bg="#f0f0f0")

input_frame.pack(pady=10)

entries = []

for i in range(10):

    lbl = tk.Label(

        input_frame,

        text=f"Value {i+1}",

        bg="#f0f0f0"
    )

    lbl.grid(row=i, column=0, padx=5, pady=5)

    entry = tk.Entry(input_frame, width=20)

    entry.grid(row=i, column=1, padx=5, pady=5)

    entries.append(entry)

# ---------------- PREDICT BUTTON ---------------- #

predict_btn = tk.Button(

    root,

    text="Predict Transaction",

    command=predict_transaction,

    bg="red",

    fg="white",

    font=("Arial", 14)
)

predict_btn.pack(pady=20)

# ---------------- RUN GUI ---------------- #

root.mainloop()