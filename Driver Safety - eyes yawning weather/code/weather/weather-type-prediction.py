#!/usr/bin/env python
# coding: utf-8

# ---
# 
# <h1 style="text-align: center;font-size: 40px;">Weather Type Prediction</h1>
# 
# ---
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import cv2
import tqdm as tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


root_dir = "dataset/"
os.listdir(root_dir)


# In[3]:


foggy = "dataset/foggy"
sunrise = "dataset/sunrise"
shine = "dataset/shine"
rainy = "dataset/rainy"
cloudy = "dataset/cloudy"
test = "dataset/alien_test"


# In[4]:


print("Number of Images in Each Directory:")
print(f"Foggy: {len(os.listdir(foggy))}")
print(f"Sunrise: {len(os.listdir(sunrise))}")
print(f"Shine: {len(os.listdir(shine))}")
print(f"Rainy: {len(os.listdir(rainy))}")
print(f"Cloudy: {len(os.listdir(cloudy))}")


# <h3>Data preprocessing for our Model</h3>

# In[5]:


x = []
y = [] 
dataset =[]
def create_dataset(directory,dir_name):
    for i in tqdm.tqdm(os.listdir(directory)):
        full_path = os.path.join(directory,i)
        try:
            img = cv2.imread(full_path)
            img = cv2.resize(img,(150,150))
        except:
            continue
        x.append(img)
        y.append(dir_name)
    return x,y


# In[6]:


x,y= create_dataset(foggy,"foggy")
x,y= create_dataset(sunrise,"sunrise")
x,y= create_dataset(shine,"shine")
x,y= create_dataset(rainy,"rainy")
x,y= create_dataset(cloudy,"cloudy")


# In[7]:


x =  np.array(x)
y = np.array(y)
x.shape,y.shape


# In[8]:


'''import seaborn as sns
plt.figure(figsize=(9,7))
plt.style.use("fivethirtyeight")
y_counts = y.value_counts()
sns.countplot(y)
plt.show()'''


# ><h3>Let's see some of the Images</h3>

# In[9]:


fig = plt.figure(figsize=(12,7))
for i in range(15):
    sample =  random.choice(range(len(x)))
    image = x[sample]
    category = y[sample]
    plt.subplot(3,5,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image)
    plt.xlabel(category)
    
plt.tight_layout()
plt.show()


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[12]:


img_size =150

x_train = np.array(x_train)/255.0
x_test = np.array(x_test)/255.0


x_train = x_train.reshape(-1,img_size,img_size,3)
y_train = np.array(y_train)

x_test = x_test.reshape(-1,img_size,img_size,3)
y_test = np.array(y_test)


# In[13]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train_lb = lb.fit_transform(y_train)
y_test_lb = lb.fit_transform(y_test)


# In[14]:


y_train_lb.shape,y_test_lb.shape


# ><h3>Model building</h3>

# In[15]:


from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights = "imagenet",include_top=False,input_shape=(img_size,img_size,3))


# In[16]:


for layer in vgg.layers:
    layer.trainable = False


# In[17]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
model =Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(5,activation="softmax"))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_accuracy",verbose=1,save_best_only=True,
                             save_weights_only=False)
earlystop = EarlyStopping(monitor="val_accuracy",patience=5,verbose=1)


# In[20]:


unique,counts = np.unique(y_train_lb,return_counts=True)
print(unique,counts)


# In[21]:


batch_size=32
history =  model.fit(x_train,y_train_lb,epochs=15,validation_data=(x_test,y_test_lb),
                     batch_size=32 ,verbose=1,callbacks=[checkpoint,earlystop])


# In[22]:


loss,accuracy = model.evaluate(x_test,y_test_lb)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")     


# In[24]:


import numpy as np

# Assuming model is your trained Sequential model
y_probabilities = model.predict(x_test)
y_pred = np.argmax(y_probabilities, axis=1)

# Display the first 15 predictions
print(y_pred[:15])


# ><h3>Classification Report</h3>

# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ><h3>Confusion Matrix</h3>

# In[27]:


get_ipython().system('pip install mlxtend')


# In[28]:


from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["cloudy","foggy","rainy","shine","sunrise"],
                      show_normed = True);


# ><h3>Learning Curve:</h3>

# In[31]:


plt.style.use("ggplot")
fig = plt.figure(figsize=(12,6))
epochs = range(1,16)
plt.subplot(1,2,1)
plt.plot(epochs[:len(history.history["accuracy"])], history.history["accuracy"], "go-")
plt.plot(epochs[:len(history.history["val_accuracy"])], history.history["val_accuracy"], "ro-")

plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","val"],loc = "upper left")

plt.subplot(1,2,2)
plt.plot(epochs[:len(history.history["loss"])], history.history["loss"], "go-")
plt.plot(epochs[:len(history.history["val_loss"])], history.history["val_loss"], "ro-")

plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train","val"],loc = "upper left")
plt.show()


# ><h3>Model Performance:</h3>

# In[32]:


plt.figure(figsize=(12,9))
plt.style.use("ggplot")
for i in range(10):
    sample = random.choice(range(len(x_test)))
    plt.subplot(2,5,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(x_test[sample])
    plt.xlabel(f"Actual: {y_test[sample]}\n Predicted: {y_pred[sample]}")
    
plt.tight_layout()    
plt.show()                       


# In[33]:


model.save("wheather_prediction.h5")


# In[ ]:




