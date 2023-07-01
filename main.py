# %%
# Import libraries
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# %%
#import dataset mnist
mnist = tf.keras.datasets.mnist

# %%
#splitting into training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %% [markdown]
# **Normatize input features(pixels which will be in range 0-255)**

# %%
x_train = tf.keras.utils.normalize(x_train, axis=1)

# %%
x_test = tf.keras.utils.normalize(x_test, axis=1)

# %%
#model object creation
model = tf.keras.models.Sequential()

# %% [markdown]
# **Adding layers to model**

# %%
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # linear straight 28x28 sized pixels
model.add(tf.keras.layers.Dense(128, activation='relu')) # First layer relu = rectify linear unit(0:-ve and then straight up) 
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Last layer (adding up results of all the first 10 neurons)

# %%
#Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
#Train the model
model.fit(x_train, y_train, epochs=7)

# %%
model.save('handwritten.model')

# %%
#load model
model = tf.keras.models.load_model('handwritten.model')

# %% [markdown]
# **Predictions**

# %%
loss, accuracy = model.evaluate(x_test, y_test)

# %%
# getting images using os
img_number = 1
while os.path.isfile(f"digits/digit{img_number}.png"):
    try: 
        img = cv2.imread(f"digits/digit{img_number}.png")[:,:, 0] # Selecting alpha values only
        imgi = np.invert(np.array([img]))
        pred = model.predict(imgi)
        print(f"{img_number}digit is probably {np.argmax(pred)}")
        plt.imshow(imgi[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        img_number += 1


