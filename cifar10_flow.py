# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D,Activation
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img




#Eğitim ve  test veri setleri
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)





#model
#Normalizasyon yapmak ağımızın daha kolay karar vermesini sağlar
#validation ve acc i yükseltir
plt.imshow(test_images[0])
train_labels[0][0]
for i in range(0,16):
    plt.subplot(4,4, i+1)
    plt.imshow(train_images[i])
    
#normalizasyon
train_images=train_images.astype('float32')/255
test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels,100)
test_labels=to_categorical(test_labels,100)

#model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Underfitting oluyorsa dropout değerlerini fazla seçmişizdir
#ezberlemeyi azaltmanın yollarından bir tanesi de parametre sayısını azaltmak
#ağın yapısını karmaşık seçmiş olabiliriz bunu düzeltmek gerekebilir
#image data generator ile veri zenginleştirme data augmentation
#başka verilerde de bu yapılabilir.
#data augmentation

model.summary()


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6, activation='softmax'))
#softmax (olasılığı gösterir) herbir class ın oranlarını topladıgımızda toplam 1.0 oluyor
#alakasız bir veri verilirse çıkışta olasılıklar dagılır

model.summary()
#modeli derleme
#olasılıklı çıkış beklediğimiz zaman ctg_crossentropy kullanıyoruz
from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#data augmentation************************
from keras.preprocessing.image import ImageDataGenerator
train_datagen =ImageDataGenerator(
    
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

test_datagen =ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

#flow ile bırakırsan direk cifar10 dan aldığı değişkenden okur
train_generator=train_datagen.flow_from_directory(#dosyadan okur
    train_images,
    train_labels,
    target_size=(32,32),    
    batch_size=20
    )
#40.dakika dan bi izle 10.hafta
validation_generator = test_datagen.flow(
        test_images,
        test_labels,
        target_size=(32, 32),
        batch_size=20)

#validation split e 0.1 dersek training set in yüzde 10 unu validation için kullanır
#oradan harcamak yerine x test ve y test i kullanabiliriz. bunun için validation_data yı kullanacagız

#validation generator , label ları o klasörlerden çıkartır. zenginleştirme yaptıgımızda ekleriz
history=model.fit(train_images,
          train_labels,
          epochs=100,
          validation_data=(test_images,test_labels)
          )


# model.save('cifar10_model1.h5')
# models.load_model('cifar10_model1.h5')
# model.save_weights(filepath, args, kwargs)
# model.load_weights(kwargs)


import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

# =============================================================================
# History 'i kaydetmenin yolu

# import numpy as np
# np.save('history1',(acc,val_acc,loss,val_loss))
# np.load('history1.npy')
# =============================================================================


epochs = range(len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'r',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.xlabel('accurarcy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.xlabel('loss')
plt.legend()

plt.figure()





