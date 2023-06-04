import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(0)
from tensorflow import keras
import numpy as np
np.random.seed(0)
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# Specify the paths to the train and valid directories
train_directory = "E:/Downloads/Plant-Leaf-Disease-Prediction-main/Dataset/train"
valid_directory = "E:/Downloads/Plant-Leaf-Disease-Prediction-main/Dataset/valid"

# Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_directory,
    target_size=(128,128),
    batch_size=32,
    class_mode='sparse'
)

test_gen = test_datagen.flow_from_directory(
    valid_directory,
    target_size=(128,128),
    batch_size=32,
    class_mode='sparse'
)




model = keras.Sequential()

model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(128,128,3)))
model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))

model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1568,activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10,activation="softmax"))

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.summary()

ep = 10
history = model.fit_generator(train_gen,
          validation_data=test_gen,
          epochs = ep)


model.save("leaf_disease_model.h5")
model.save_weights("leaf_disease_model_weights.h5")


# plt.figure(figsize = (20,5))
# plt.subplot(1,2,1)
# plt.title("Train and Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.plot(history.history['loss'],label="Train Loss")
# plt.plot(history.history['val_loss'], label="Validation Loss")
# plt.xlim(0, 10)
# plt.ylim(0.0,1.0)
# plt.legend()

# plt.subplot(1,2,2)
# plt.title("Train and Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.plot(history.history['accuracy'], label="Train Accuracy")
# plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
# plt.xlim(0, 9.25)
# plt.ylim(0.75,1.0)
# plt.legend()
# plt.tight_layout()

# labels = []
# predictions = []
# for x,y in test_gen:
#     labels.append(list(y))
#     predictions.append(tf.argmax(model.predict(x),1))



# predictions = list(itertools.chain.from_iterable(predictions))
# labels = list(itertools.chain.from_iterable(labels))

# print("Train Accuracy  : {:.2f} %".format(history.history['accuracy'][-1]*100))
# print("Test Accuracy   : {:.2f} %".format(accuracy_score(labels, predictions) * 100))
# print("Precision Score : {:.2f} %".format(precision_score(labels, predictions, average='micro') * 100))
# print("Recall Score    : {:.2f} %".format(recall_score(labels, predictions, average='micro') * 100))

# plt.figure(figsize= (20,5))
# cm = confusion_matrix(labels, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                               display_labels=list(range(1,39)))
# fig, ax = plt.subplots(figsize=(15,15))
# disp.plot(ax=ax,colorbar= False,cmap = 'YlGnBu')
# plt.title("Confusion Matrix")
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()
