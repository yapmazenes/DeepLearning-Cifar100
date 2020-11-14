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
from keras.preprocessing import image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img


import os, shutil
#Eğitim ve  test veri setleri
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()



# =============================================================================
# Resimlerin Dosyalara Aktarılması
# =============================================================================
base_dir=os.getcwd()
base_dir=base_dir+"\\Datasets\\"

os.mkdir(base_dir)

train_dir=os.path.join(base_dir, "training")
os.mkdir(train_dir)
os.mkdir(train_dir+"\\baby")
os.mkdir(train_dir+"\\bed")
os.mkdir(train_dir+"\\mouse")
os.mkdir(train_dir+"\\pear")
os.mkdir(train_dir+"\\snail")
os.mkdir(train_dir+"\\wardrope")

test_dir=os.path.join(base_dir, "test")
os.mkdir(test_dir)
os.mkdir(test_dir+"\\baby")
os.mkdir(test_dir+"\\bed")
os.mkdir(test_dir+"\\mouse")
os.mkdir(test_dir+"\\pear")
os.mkdir(test_dir+"\\snail")
os.mkdir(test_dir+"\\wardrope")


for i in range(0,50000):
    if train_labels[i][0] == 2 :
        img.imsave(train_dir+"\\baby\\"+str(i)+".png", train_images[i])
        #baby dosyasına fotoyu kaydet  train
    elif train_labels[i][0] == 5 :
         img.imsave(train_dir+"\\bed\\"+str(i)+".png", train_images[i])
        #bed dosyasına fotoyu kaydet
    elif train_labels[i][0] == 50 :
        img.imsave(train_dir+"\\mouse\\"+str(i)+".png", train_images[i])
        #=mouse  dosyasına fotoyu kaydet
    elif train_labels[i][0] == 57 :
        img.imsave(train_dir+"\\pear\\"+str(i)+".png", train_images[i])
        #pear dosyasına fotoyu kaydet
    elif train_labels[i][0] == 77 :
        img.imsave(train_dir+"\\snail\\"+str(i)+".png", train_images[i])
        #snail dosyasına fotoyu kaydet
    elif train_labels[i][0] == 94 :
        img.imsave(train_dir+"\\wardrope\\"+str(i)+".png", train_images[i])
        #wardrope dosyasına fotoyu kaydet
        
        

        


#Test datalarının kaydedilmesi
        
for i in range(0,10000):
    if test_labels[i][0] == 2 :
        img.imsave(test_dir+"\\baby\\"+str(i)+".png", test_images[i])
        #baby dosyasına fotoyu kaydet  test
    elif test_labels[i][0] == 5 :
         img.imsave(test_dir+"\\bed\\"+str(i)+".png", test_images[i])
        #bed dosyasına fotoyu kaydet
    elif test_labels[i][0] == 50 :
        img.imsave(test_dir+"\\mouse\\"+str(i)+".png", test_images[i])
        #=mouse  dosyasına fotoyu kaydet
    elif test_labels[i][0] == 57 :
        img.imsave(test_dir+"\\pear\\"+str(i)+".png", test_images[i])
        #pear dosyasına fotoyu kaydet
    elif test_labels[i][0] == 77 :
        img.imsave(test_dir+"\\snail\\"+str(i)+".png", test_images[i])
        #snail dosyasına fotoyu kaydet
    elif test_labels[i][0] == 94 :
        img.imsave(test_dir+"\\wardrope\\"+str(i)+".png", test_images[i])
        #wardrope dosyasına fotoyu kaydet


# =============================================================================
# //////////////////////Resimlerin Dosyalara Aktarılması//////////////////////
# =============================================================================

#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)


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
#train_datagen =ImageDataGenerator(rescale=1./255)
test_datagen =ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(#dosyadan okur
    train_dir,
    target_size=(32,32),
    batch_size=20,
    class_mode='categorical'
    )

validation_generator = test_datagen.flow_from_directory(#dosyadan okur
        test_dir,
        target_size=(32, 32),
        batch_size=20,
        class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#model
#Normalizasyon yapmak ağımızın daha kolay karar vermesini sağlar
#validation ve acc i yükseltir



#model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.25)) 
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.25))

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
#olasılıklı bir çıkış beklendiği zaman ctg_crossentropy kullanıyoruz
from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])




#validation split e 0.1 dersek training set in yüzde 10 unu validation için kullanır
#oradan harcamak yerine x test ve y test i kullanabiliriz. bunun için validation_data yı kullanacagız

#validation generator , label ları o klasörlerden çıkartır. zenginleştirme yaptıgımızda ekleriz
#Sonraki

history=model.fit_generator(train_generator,
          steps_per_epoch=600,
          epochs=60,
          validation_data=validation_generator,
          validation_steps=30
          )
model.save('cifar100-ImageAug')

#Önceki
history=model.fit_generator(train_generator,
          steps_per_epoch=150,
          epochs=100,
          validation_data=validation_generator,
          validation_steps=30
          )
model.save('cifar100-ImageAug')

# model.save('cifar10_model1.h5')
#model = models.load_model('cifar100-ImageAug')
# model.save_weights(filepath, args, kwargs)
# model.load_weights(kwargs)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

print(val_acc)

# =============================================================================
# History 'i kaydetmenin yolu

import numpy as np
# np.save('history1',(acc,val_acc,loss,val_loss))
# (acc,val_acc,loss,val_loss)=np.load('historyImageAug.npy')
# =============================================================================
np.save('historyImageAug',(acc,val_acc,loss,val_loss))

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc ( Eğitim doğruluğu )')
plt.plot(epochs,val_acc,'r',label='Validation acc ( Onaylama doğruluğu)')
plt.title('Training and validation accuracy \n( Eğitim ve onaylama doğruluğu)')
plt.xlabel('epochs ( Epoklar )')
plt.xlabel('accurarcy ( Doğruluk )')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss ( Eğitim Kayıpları )')
plt.plot(epochs,val_loss,'r',label='Validation loss ( Doğrulama Kayıpları )')
plt.title('Training and validation loss \n( Eğitim ve doğrulama kayıpları )')
plt.xlabel('epochs ( Epoklar )')
plt.xlabel('loss ( Kayıplar )')
plt.legend()

plt.figure()





# =============================================================================
# predict dediğimiz zaman tüm katmanlardan geçerek bir sonuç üretir, bu sebeple doğru sonucu üretebilme 
# aşamasında , başarımı arttırma eğiliminde ez az sayıda katman ile başarımın elde edilmesi
 # bizim için çok daha iyidir. Parametre sayısı adedince çarpma işlemi olacaktır ve bununda bir karmaşıklığı olacaktır.
# =============================================================================

# W=model.get_weights()

# W[0].shape  # ilk katmanın ağırlıkları 
# ilk katmanda 32 tane filtreleme işlemimiz var. ve ilk başta sonuç (3,3,3,32)
# neden 3 e 3 ten 3 tane var çünkü ilk katmanda rgb ler de ayrı filtreleniyor

# W[1].shape
# daha sonraki katmanlarda, mesela 1 'de' bias ile ilgili olanda shape 32 gelir
# W[2].shape 
# burda ise RGB 'birleştiği için kanallar ayrı ayrı olmak yerine birlikte ilerliyor.
# dolayısıyla 3 e 3 lük 32  ilk katmanın 32 çıkışı , 2.katmanını 32 girişi var dolayısıyla herbiri 32 taneye dağıldığı için burda filtre sayısı 32x32 ye çıkmış oluyor
# W2=W[2]
# W2[:,:,0,0,] #KERNEL 'lardan bir tanesinin kat sayıları
# eğitilmiş ağımızda görüntüyü filtreleyen kısım
# 32 32 tane çıkışlardan diğerinin girişine giden ağırlıklardan bir tanesi
# onların 0 a 0 konumundaki  W2[:,:,0,0,]
# veya 0 dan 1 e giden 2 ye 3 e giden ağırlıklar W2[:,:,0,1,] W2[:,:,0,2,] W2[:,:,0,3,]
# 1.çıkıştan 3 e giden ağırlıklar gibi gibi W2[:,:,1,3,]
# herbiri farklı özellikleri seçtiği için katsayıları da farklı

# =============================================================================
# Aktivasyon 'ların Görselleştirilmesi
# =============================================================================

from keras.models import load_model

model = load_model('C:\\Users\\Enes Yapmaz\\Desktop\\enes\\dl\\Odev\\DropOutTrainValues\\cifar100-Dropout')
model.summary()  # As a reminder.

img_path = 'Datasets/test/baby/887.png'

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(32, 32))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]


print(first_layer_activation.shape)


y=model.predict(img_tensor)
y.max()
y.argmax()



import matplotlib.pyplot as plt

for i in range(32):
    plt.subplot(4,8, i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    
# =============================================================================
# Aktivasyon 'ların Görselleştirilmesi
# =============================================================================




# =============================================================================
# #Samples of ImageAug
# =============================================================================

fname="c.png"

fnames = [os.path.join(train_dir+"\\baby", fname) for fname in os.listdir(train_dir+"\\baby")]

# We pick one image to "augment"

s = (16,32,32,3)
arr=np.empty(s)
def show(k,arrNumber):
    
    img_path = fnames[k]
# Read the image and resize it
    img = image.load_img(img_path, target_size=(32, 32))

# Convert it to a Numpy array with shape (32, 32, 3)
    x = image.img_to_array(img)

# Reshape it to (1, 32, 32, 3)
    x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
    i = arrNumber
    for batch in train_datagen.flow(x, batch_size=1):
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        
        arr[i]=batch[0]
        # arrs[i]=arr
        i += 1
        if i % 4== 0:
            break
        
arrNumber=0
for i in range(4):
    show(i,arrNumber)
    arrNumber+=4
    
    
for i in range(16):
    plt.subplot(4,4, i+1)
    plt.imshow(arr[i])

# =============================================================================
# #Samples of ImageAug
# =============================================================================




# =============================================================================
# #Samples of All image class
# =============================================================================
fname="c.png"
shape = (30,32,32,3)
arrSample=np.empty(shape)
def getImage(folder,imgNum):

    fnames = [os.path.join(train_dir+"\\"+folder, fname) for fname in os.listdir(train_dir+"\\"+folder)]

    img_path = fnames[imgNum]
# Read the image and resize it
    img = image.load_img(img_path, target_size=(32, 32))

# Convert it to a Numpy array with shape (32, 32, 3)
    x = image.img_to_array(img)
    print(x)
    return x

for i in range (0,5):
    arrSample[i] = getImage("baby",i)
    
for i in range (5,10):
    arrSample[i] = getImage("bed",i)
for i in range (10,15):
    arrSample[i] = getImage("mouse",i)
for i in range (15,20):
    arrSample[i] = getImage("pear",i)
for i in range (20,25):
    arrSample[i] = getImage("snail",i)
    
for i in range (25,30):
    arrSample[i] = getImage("wardrope",i)

plt.figure(arrSample[0])

for i in range(30):
    plt.subplot(6,5, i+1)
    plt.imshow(image.array_to_img(arrSample[i]))
    # =============================================================================
# #Samples of All image class
# =============================================================================