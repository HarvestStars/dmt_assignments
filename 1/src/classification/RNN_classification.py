# RNN classification code
# import lots of lots of libraries before you start...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
Output list:
(1) plot of loss (train vs. validation)
(2) plot of accuracy (train vs. validation)
(3) train and test accuracy
(4) confusion matrix
"""

"""
Citation for the code:https://www.kaggle.com/code/szaitseff/classification-of-time-series-with-lstm-rnn
"""

# step 1: load the data first
train_df = pd.read_csv("./source_data/train_split_by_id.csv")
test_df = pd.read_csv("./source_data/test_split_by_id.csv")

# step 2: group the predictors and targets
features = [col for col in train_df.columns if col.endswith("_hist_mean")]
target = "mood_type"

# step 3: transform the data and divide the dataset into training and testing sets
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# remove NaN values with mean imputation
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# step 4: convert data for LSTM
X_train_lstm = np.expand_dims(X_train_scaled, axis=1)
X_test_lstm = np.expand_dims(X_test_scaled, axis=1)
LAYERS = [8, 8, 8, 1]

# step 5: build the RNN model
model1 = Sequential()
model1.add(InputLayer(input_shape=(1, X_train.shape[1])))
model1.add(
    LSTM(
        units=LAYERS[0],
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_regularizer=l2(3e-2),
        recurrent_regularizer=l2(3e-2),
        return_sequences=True,
    )
)
model1.add(BatchNormalization())

model1.add(
    LSTM(
        units=LAYERS[1],
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_regularizer=l2(3e-2),
        recurrent_regularizer=l2(3e-2),
        return_sequences=True,
    )
)
model1.add(BatchNormalization())

model1.add(
    LSTM(
        units=LAYERS[2],
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        kernel_regularizer=l2(3e-2),
        recurrent_regularizer=l2(3e-2),
        return_sequences=False,
    )
)
model1.add(BatchNormalization())
model1.add(
    Dense(units=LAYERS[3], activation="sigmoid")
)  # Dense layer and sigmoid activation
print(model1.summary())

# step 6: compile the model with the lowest loss
cp = ModelCheckpoint("model1.h5", save_best_only=True)
model1.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

History = model1.fit(
    X_train_lstm,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    callbacks=[cp],
)

# step 7: load the best model and evaluate it
model1 = load_model("model1.h5")
train_predictions = model1.predict(X_train_lstm).flatten()

# evaluate loss and accuracy
train_loss, train_accuracy = model1.evaluate(X_train_lstm, y_train, verbose=0)
test_loss, test_accuracy = model1.evaluate(X_test_lstm, y_test, verbose=0)
print(f"train accuracy = {round(train_accuracy * 100, 4)}%")
print(f"test accuracy = {round(test_accuracy * 100, 4)}%")

# step 8: plot curves for loss and accuracy
plt.figure(figsize=(10, 5))
plt.plot(History.history["loss"], label="train_loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.title("Loss curve for training and validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(History.history["accuracy"], label="train_accuracy")
plt.plot(History.history["val_accuracy"], label="val_accuracy")
plt.title("Accuracy curve for training and validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# step 9: plot confusion matrix
test_predictions = model1.predict(X_test_lstm).flatten()
y_pred = (test_predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Class 0", "Class 1"]
)
disp.plot(cmap="Blues", values_format="d")  # 'd' for integer display
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
