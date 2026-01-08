import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# ---- Step 1: Load .npy Data Files ----
is_init = False
label = []
dictionary = {}
c = 0

# Load all .npy files in the folder (except labels.npy)
for filename in os.listdir():
    if filename.endswith(".npy") and filename != "labels.npy":
        data = np.load(filename)

        if not is_init:
            X = data
            size = data.shape[0]
            y = np.array([filename.split('.')[0]] * size).reshape(-1, 1)
            is_init = True
        else:
            X = np.concatenate((X, data))
            size = data.shape[0]
            y = np.concatenate((y, np.array([filename.split('.')[0]] * size).reshape(-1, 1)))

        label.append(filename.split('.')[0])
        dictionary[filename.split('.')[0]] = c
        c += 1

# ---- Step 2: Encode Labels ----
# Convert string labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode labels
y = to_categorical(y)

# ---- Step 3: Shuffle Data ----
X_new = np.empty_like(X)
y_new = np.empty_like(y)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

for i, idx in enumerate(indices):
    X_new[i] = X[idx]
    y_new[i] = y[idx]

# ---- Step 4: Define the Model ----
input_shape = (X.shape[1],)  # ✅ Fix: Input shape must be a tuple
ip = Input(shape=input_shape)

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# ---- Step 5: Train the Model ----
model.fit(X_new, y_new, epochs=50, batch_size=32)

# ---- Step 6: Save the Model and Labels ----
model.save("model.h5")
np.save("labels.npy", np.array(label))

print("✅ Training complete. Model saved as 'model.h5' and labels as 'labels.npy'")
