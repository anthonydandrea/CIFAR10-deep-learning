
# coding: utf-8

# In[1]:


import numpy as np
from numpy import array


# In[2]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout
from keras.layers import Activation, Flatten, MaxPooling2D


# In[3]:


from keras.utils import np_utils


# In[4]:


from keras.datasets import cifar10

(X_train,y_train), (X_test,y_test) = cifar10.load_data()


# In[5]:


print(X_train.shape)


# In[ ]:


New_X_train = []
for x in range(len(X_train)):
    New_X_train.append([])
    for y in range(len(X_train[0])-1,-1,-1):
        New_X_train[x].append(X_train[x][y])
    
New_X_train = array(New_X_train)
print(len(New_X_train[0]))


# In[ ]:


get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
New_X_train = np.fliplr(X_train)


# In[ ]:


from keras import backend
backend.set_image_dim_ordering('th')


# In[ ]:


print(X_train.shape)
print(type(X_train))
print(type(New_X_train))


# In[ ]:


X_train = np.concatenate((X_train, New_X_train))
y_train = np.concatenate((y_train, y_train))


# In[ ]:


print(len(X_train))
print(len(y_train))


# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


# In[ ]:


print(y_train[:10])
print(type(X_train))


# In[ ]:


Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

print(Y_train[:10])


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Conv2D(32,(3,3),activation='relu',input_shape=(3,32,32)))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())


# In[ ]:


model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

print(model.summary())


# In[ ]:


model.fit(X_train,Y_train,batch_size=32,epochs=3,verbose=1)


# In[ ]:


score = model.evaluate(X_test,Y_test,verbose=1)


# In[ ]:


score

