import tensorflow as tf
from tensorflow.keras.preprocessing import image
#from keras.preprocessing import image
import numpy as np

img_width,img_height=125,125
#def test(fln):
model=tf.keras.models.load_model(r'SAvedModel path.h5')
#D:\DATASET\Covid19-dataset\test\Normal\0102.jpeg
sample=r'your path for the testing image'
print(sample)
#0 label for covid pics and 1 label for nromal pic
test_img=image.load_img(sample,target_size=(img_width,img_height))
    #test_img=image.load_img(sample,target_size=(img_width,img_height))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
#test_img=test_img/255
result=model.predict(test_img)
#return(result)
print(result)


# D:\hard.png
# D:\Adobe Photoshop CC 2018\iphone 8plus\J5\1.jpg
#D:\DATASET\Covid19-dataset\test\Viral Pneumonia
#D:\DATASET\Covid19-dataset\test\Normal
