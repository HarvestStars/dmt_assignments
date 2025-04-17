# RNN trial code
# import lots of lots of libraries before you start...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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
(5) plot of the residuals foe test set
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

# step 4: convert data for LSTM
X_train_lstm = np.expand_dims(X_train, axis=1)  # (samples, 1, features)
X_test_lstm = np.expand_dims(X_test, axis=1)

# step 5: build the RNN model
model1 = Sequential()
model1.add(InputLayer(input_shape=(1, X_train.shape[1])))
model1.add(LSTM(64, return_sequences=True))  # LSTM layer with 64 units
model1.add(Dense(32, activation="relu"))  # Dense layer and ReLU activation
model1.add(Dense(1, activation="linear"))  # Output layer with linear activation
model1.summary()

# step 6: compile the model with the lowest MSE loss
cp = ModelCheckpoint("model1.h5", save_best_only=True)
model1.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=MeanSquaredError(),
    metrics=[RootMeanSquaredError()],
)
model1.fit(X_train_lstm, y_train, epochs=100, validation_split=0.1, callbacks=[cp])

# step 7: load the best model and evaluate it
model1 = load_model("model1.h5")
train_predictions = model1.predict(X_train_lstm).flatten()
print(train_predictions.shape, y_train.flatten().shape)
train_results = pd.DataFrame(
    data={"train_predictions": train_predictions, "Actuals": y_train.flatten()}
)
train_results["residuals"] = (
    train_results["train_predictions"] - train_results["Actuals"]
)
# calculate the mean squared error and root mean squared error
mse_tra = mean_squared_error(
    train_results["Actuals"], train_results["train_predictions"]
)
rmse_tra = np.sqrt(mse_tra)

train_results.to_csv("train_results.csv", index=False)  # save training results to a csv

# plot the difference between the predictions and the actuals
plt.plot(train_results["train_predictions"], label="Predictions")
plt.plot(train_results["Actuals"], label="Actuals")
plt.legend()
plt.grid()
plt.title("Training Results")
plt.show()

# step 8: evaluate the model on the test result
test_predictions = model1.predict(X_test_lstm).flatten()
test_results = pd.DataFrame(
    data={"test_predictions": test_predictions, "Actuals": y_test.flatten()}
)
test_results["residuals"] = test_results["test_predictions"] - test_results["Actuals"]

# calculate the mean squared error and root mean squared error
mse = mean_squared_error(test_results["Actuals"], test_results["test_predictions"])
rmse = np.sqrt(mse)

test_results.to_csv("test_results.csv", index=False)  # save results to a csv

# plot the difference between the predictions and the actuals
plt.plot(test_results["test_predictions"], label="Predictions", marker="o")
plt.plot(test_results["Actuals"], label="Actuals", marker="x")
plt.legend()
plt.grid()
plt.title("Testing Results")
plt.show()

# plot the difference in residuals
plt.plot(test_results["residuals"], label="Testing Residuals", color="red")
plt.legend()
plt.grid()
plt.title("Residuals")
plt.show()

print("MSE of training:", mse_tra, "rMSE of training", rmse_tra)
print("MSE of testing:", mse, "rMSE of testing", rmse)
