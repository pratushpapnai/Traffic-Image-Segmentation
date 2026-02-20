import utils
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

BATCH_SIZE=500
unet_model=load_model("C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/model/ImageSegmenter.keras")

TEST_IMAGE_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/test/"
TEST_MASK_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/test_labels/"

test_image_org=os.listdir(TEST_IMAGE_PATH)
test_mask_org=os.listdir(TEST_MASK_PATH)

test_image_list=[TEST_IMAGE_PATH+i for i in test_image_org]
test_mask_list=[TEST_MASK_PATH+i for i in test_mask_org]
test_image_path=tf.constant(test_image_list)
test_mask_path=tf.constant(test_mask_list)

test_dataset=tf.data.Dataset.from_tensor_slices((test_image_path,test_mask_path))

test_ds=test_dataset.map(utils.process_path)
processed_test_ds=test_ds.map(utils.preprocess)

print(processed_test_ds)
for img,mask in test_dataset.take(1):
    print(img)
    print(mask)
    
test_dataset=processed_test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for img_batch, mask_batch in test_dataset.take(2):
    print(img_batch.shape, mask_batch.shape)
    
def show_pred(image_path,mask_path,model):
    img,mask=utils.process_path(image_path,mask_path)
    img,class_mask=utils.preprocess(img,mask)

    img=img.numpy()
    class_mask=class_mask.numpy()

    img=np.expand_dims(img,axis=0)
    prediction=model.predict(img,verbose=0)
    img=img[0]
    prediction=prediction[0]
    prediction=np.argmax(prediction,axis=-1)
    
    print(class_mask.shape)
    
    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(class_mask)
    plt.title("Actual Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(prediction)
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()
    
show_pred(test_image_path[200],test_mask_path[200],unet_model)