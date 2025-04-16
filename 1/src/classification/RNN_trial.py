# RNN trial code
# import lots of lots of libraries before you start...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.models import load_model

"""
Output list:
(1) csv file with the training results
(2) csv file with the testing results
(3) plot of the training results
(4) plot of the testing results
"""

# step 1: load the data first
filepath = "../../raw_data/cleaned_data_daily_summary_mood_imputed_sliding_window.csv"
df = pd.read_csv(filepath)
df.fillna(0, inplace=True)

# step 2: group the predictors and targets
features = [
    "mood_hist_mean",
    "activity_hist_mean",
    "appCat.builtin_hist_mean",
    "appCat.office_hist_mean",
    "call_hist_mean",
    "circumplex.arousal_hist_mean",
    "circumplex.valence_hist_mean",
    "screen_hist_mean",
]
X = df[features].values
y = df["mood_target"].values

# step 3: transform the data and divide the dataset into training and testing sets
scaler = StandardScaler()  # standardize the data
X_scaled = scaler.fit_transform(X)  # fit the scaler to the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.05, train_size=0.95, random_state=42
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 4. Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = np.expand_dims(X_train, axis=1)  # (samples, 1, features)
X_test_lstm = np.expand_dims(X_test, axis=1)

# step 5: build the RNN model
model1 = Sequential()
model1.add(InputLayer(input_shape=(X_train.shape[1], 1)))
model1.add(LSTM(64, return_sequences=True))  # LSTM layer with 64 units
model1.add(Dense(32, activation="relu"))  # Dense layer and ReLU activation
model1.add(Dense(1, activation="linear"))  # Output layer with linear activation
model1.summary()

# step 6: compile the model with the lowest MSE loss
cp = ModelCheckpoint("model1/", save_best_only=True)
model1.compile(
    optimizer='adam',
    loss=MeanSquaredError(),
    metrics=[RootMeanSquaredError()],
)

# 7. Train the model
history = model1.fit(
    X_train_lstm, y_train, epochs=100, validation_split=0.2, callbacks=[cp]
)

# 8. Load the best model and evaluate
model1 = load_model("model1.keras")
train_predictions = model1.predict(X_train_lstm).flatten()
train_results = pd.DataFrame({
    "train_predictions": train_predictions,
    "Actuals": y_train.flatten(),
})
train_results.to_csv("train_results.csv", index=False)

plt.plot(train_results["train_predictions"], label="Predictions")
plt.plot(train_results["Actuals"], label="Actuals")
plt.legend()
plt.title("Training Results")
plt.show()

# 9. Test evaluation
test_predictions = model1.predict(X_test_lstm).flatten()
test_results = pd.DataFrame({
    "test_predictions": test_predictions,
    "Actuals": y_test.flatten(),
})
test_results.to_csv("test_results.csv", index=False)

plt.plot(test_results["test_predictions"], label="Predictions")
plt.plot(test_results["Actuals"], label="Actuals")
plt.legend()
plt.title("Testing Results")
plt.show()