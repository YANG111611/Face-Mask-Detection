import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout)
from keras.callbacks import (EarlyStopping, ModelCheckpoint)

DATA_PATH = "data"
IMG_SIZE = 128

X = []
y = []

classes = ['with_mask', 'without_mask']

for label, class_name in enumerate(classes):
    class_path = os.path.join(DATA_PATH, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        X.append(image)
        y.append(label)

X = np.array(X)
y = np.array(y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    shuffle = True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size = 0.2,
    random_state = 42,
    shuffle = True
)

model = Sequential([
    Input(shape = (128, 128, 3)),
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Flatten(),
    Dense(units = 128, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 1, activation = 'sigmoid')
])

model.summary()
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
print("Model compiled successfully.")

early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True,
    verbose = 1
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor = "val_loss",
    save_best_only = True,
    mode = "min",
    verbose = 1
)

history = model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    epochs = 20,
    batch_size = 32,
    callbacks = [early_stopping, checkpoint],
    verbose = 1
)

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.show()

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose = 1
)
print(f"Test Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss : {test_loss:.4f}")


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
print(classification_report(
    y_test,
    y_pred,
    target_names=["With Mask","Without Mask"]
))