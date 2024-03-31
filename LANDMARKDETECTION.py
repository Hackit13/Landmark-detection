#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt
import os
import random
from PIL import Image


# In[5]:


samples = 20000
df = pd.read_csv("train.csv")
df = df.loc[:samples,:]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)


# In[6]:


print("Size of training data:", df.shape)
print("Number of unique classes:", num_classes)


# In[7]:


data = pd.DataFrame(df['landmark_id'].value_counts())
#index the data frame
data.reset_index(inplace=True) 
data.columns=['landmark_id','count']

print(data.head(10))
print(data.tail(10))


# In[8]:


print(data['count'].describe())#statistical data for the distribution
plt.hist(data['count'],100,range = (0,57),label = 'test')#Histogram of the distribution
plt.xlabel("Amount of images")
plt.ylabel("Occurences")


# In[9]:


print("Amount of classes with five and less datapoints:", (data['count'].between(0, 5)).sum())
print("Amount of classes with between five and 10 datapoints:", (data['count'].between(5, 10)).sum())

n = plt.hist(data["landmark_id"], bins=data["landmark_id"].nunique())
freq_info = n[0]

plt.xlim(0, data['landmark_id'].max())
plt.ylim(0, data['count'].max())
plt.xlabel('Landmark ID')
plt.ylabel('Number of images')
plt.title('Histogram of Image Counts per Landmark ID')
plt.show()


# In[10]:


import os
import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tarfile

# Replace these paths with the correct paths to your data
base_path = r"C:\Users\ASUS\Desktop\Landmark detection project\images_000.tar"
extracted_folder = r"C:\Users\ASUS\Desktop\Landmark detection project\extracted_images"

# Extract the contents of the tar file
with tarfile.open(base_path, 'r') as tar:
    tar.extractall(path=extracted_folder)

# Sample code to read and display random images
num_samples = 4
fig = plt.figure(figsize=(16, 16))

for i in range(1, num_samples + 1):
    # Choose a random subfolder
    random_folder = os.path.join(extracted_folder, random.choice(os.listdir(extracted_folder)))

    # Choose a random image file from the subfolder
    random_img = os.path.join(random_folder, random.choice(os.listdir(random_folder)))

    # Read and display the image
    img = np.array(Image.open(random_img))
    fig.add_subplot(1, num_samples, i)
    plt.imshow(img)
    plt.axis('off')

plt.show()



# In[ ]:


from keras.applications import VGG19
from keras.layers import *
from keras import Sequential

### Parameters
# learning_rate   = 0.0001
# decay_speed     = 1e-6
# momentum        = 0.09

# loss_function   = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
#new_layer = Dense(num_classes, activation=activations.softmax, name='prediction')
drop_layer = Dropout(0.5)
drop_layer2 = Dropout(0.5)


model = Sequential()
for layer in source_model.layers[:-1]: # go through until last layer
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
#     if layer == source_model.layers[-3]:
#         model.add(drop_layer)
# model.add(drop_layer2)
model.add(Dense(num_classes, activation="softmax"))
model.summary()


opt1 = keras.optimizers.RMSprop(learning_rate = 0.0001, momentum = 0.09)
opt2 = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=opt1,
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

#sgd = SGD(lr=learning_rate, decay=decay_speed, momentum=momentum, nesterov=True)
# rms = keras.optimizers.RMSprop(lr=learning_rate, momentum=momentum)
# model.compile(optimizer=rms,
#               loss=loss_function,
#               metrics=["accuracy"])
# print("Model compiled! \n")


# In[ ]:


### Function used for processing the data, fitted into a data generator.
def get_image_from_number(num, df):
    fname, label = df.iloc[num,:]
    fname = fname + ".jpg"
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(base_path,path))
    return im, label

def image_reshape(im, target_size):
    return cv2.resize(im, target_size)
    
def get_batch(dataframe,start, batch_size):
    image_array = []
    label_array = []
    
    end_img = start+batch_size
    if end_img > len(dataframe):
        end_img = len(dataframe)

    for idx in range(start, end_img):
        n = idx
        im, label = get_image_from_number(n, dataframe)
        im = image_reshape(im, (224, 224)) / 255.0
        image_array.append(im)
        label_array.append(label)
        
    label_array = encode_label(label_array)
    return np.array(image_array), np.array(label_array)
batch_size = 16
epoch_shuffle = True
weight_classes = True
epochs = 15

# Split train data up into 80% and 20% validation
train, validate = np.split(df.sample(frac=1), [int(.8*len(df))])
print("Training on:", len(train), "samples")
print("Validation on:", len(validate), "samples")

    
for e in range(epochs):
    print("Epoch: ", str(e+1) + "/" + str(epochs))
    if epoch_shuffle:
        train = train.sample(frac = 1)
    for it in range(int(np.ceil(len(train)/batch_size))):

        X_train, y_train = get_batch(train, it*batch_size, batch_size)

        model.train_on_batch(X_train, y_train)
        

model.save("Model.h5")


# In[ ]:


### Test on training set
batch_size = 16

errors = 0
good_preds = []
bad_preds = []

for it in range(int(np.ceil(len(validate)/batch_size))):

    X_train, y_train = get_batch(validate, it*batch_size, batch_size)

    result = model.predict(X_train)
    cla = np.argmax(result, axis=1)
    for idx, res in enumerate(result):
        print("Class:", cla[idx], "- Confidence:", np.round(res[cla[idx]],2), "- GT:", y_train[idx])
        if cla[idx] != y_train[idx]:
            errors = errors + 1
            bad_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])
        else:
            good_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])

print("Errors: ", errors, "Acc:", np.round(100*(len(validate)-errors)/len(validate),2))

#Good predictions
good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key = lambda x: x[2], reverse=True))

fig=plt.figure(figsize=(16, 16))
for i in range(1,6):
    n = int(good_preds[i,0])
    img, lbl = get_image_from_number(n, validate)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 5, i)
    plt.imshow(img)
    lbl2 = np.array(int(good_preds[i,1])).reshape(1,1)
    sample_cnt = list(df.landmark_id).count(lbl)
    plt.title("Label: " + str(lbl) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(lbl) + ": " + str(sample_cnt))
    plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




