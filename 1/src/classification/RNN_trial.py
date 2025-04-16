# RNN trial code
# import lots of lots of libraries before you start...
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model

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

# step 4: convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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
    optimizer=Adam(learning_rate=0.002),
    loss=MeanSquaredError(),
    metrics=[RootMeanSquaredError()],
)
model1.fit(
    X_train_tensor, y_train_tensor, epochs=100, validation_split=0.2, callbacks=[cp]
)

# step 7: load the best model and evaluate it
model1 = load_model("model1/")
train_predictions = model1.predict(X_train_tensor).flatten()
train_results = pd.DataFrame(
    data={"train_predictions": train_predictions, "Actuals": y_train_tensor.flatten()}
)
train_results.to_csv("train_results.csv", index=False)  # save training results to a csv

# plot the difference between the predictions and the actuals
plt.plot(train_results["train_predictions"], label="Predictions")
plt.plot(train_results["Actuals"], label="Actuals")
plt.legend()
plt.title("Training Results")
plt.show()

# step 8: evaluate the model on the test result
test_predictions = model1.predict(X_test_tensor).flatten()
test_results = pd.DataFrame(
    data={"test_predictions": test_predictions, "Actuals": y_test_tensor.flatten()}
)
test_results.to_csv("test_results.csv", index=False)  # save results to a csv

# plot the difference between the predictions and the actuals
plt.plot(test_results["test_predictions"], label="Predictions")
plt.plot(test_results["Actuals"], label="Actuals")
plt.legend()
plt.title("Testing Results")
plt.show()
