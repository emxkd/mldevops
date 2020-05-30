#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np

# In[ ]:

json_file = open('/model/architecture/model_num.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# In[ ]:

loaded_model.summary()

# In[ ]:

Layers = []
for i,layer in enumerate(loaded_model.layers):
    #input_output = {'name': layer.name,'input': int(layer.input_shape[1]),'output': int(layer.output_shape[1])}
    if i == int(0):
        (_, inp) = layer.input_shape[0]
        input_output = [layer.name,inp]
    else:
        input_output = [layer.name,int(layer.input_shape[1]),int(layer.output_shape[1])]
    layer_name = 'layer_{}'.format(i)
    input_output.insert(0,layer_name)
    Layers.append(input_output)


# In[ ]:


def tweakTheModel(Layers):
    n_units = Layers[-2][-2]
    
    global x
    global inp_layer
    for l in Layers[:-1]:
        if '0' in l[0] :
            l_o = l[2]
            inp_layer = Input(shape=l_o)
            x = Dense(l_o,activation=relu)(inp_layer)
            #print(l_o,l_i,l_n,inp_layer,x)
        else:
            l_n = l[1]
            l_o = l[3]
            x = Dense(l_o,activation=relu)(x)
            #print(l_o,l_n)
    if n_units >= int(64):
        x = Dense(n_units // 2, activation=relu)(x)
    else:
        x = Dense(n_units*2, activation=relu)(x)
    x = Dense(Layers[-1][-1],activation='softmax')(x)
    return x


# In[ ]:

out = tweakTheModel(Layers)

# In[ ]:

model = Model(inputs=[inp_layer],outputs=[out])

# In[ ]:


model.summary()


# In[ ]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# In[ ]

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# In[ ]

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# In[ ]

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

# In[ ]

results = model.evaluate(x_test, y_test)

# In[ ]

p = "{}".format(results[1])

# In[ ]

with open('/results/data.txt', 'w') as f:
    f.write(p)

# In[ ]

model_json = model.to_json()
with open("/model/architecture/model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("/model/architecture/model_num.h5")

# In[ ]

model.save('/model/model_weight/model_mnist_{}.h5'.format(p))
