import cv2
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import utils
import model_utils

COLORS=utils.COLORS

TRAIN_IMAGE_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/train/"
TRAIN_MASK_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/train_labels/"
VALID_IMAGE_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/val/"
VALID_MASK_PATH="C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/val_labels/"

train_image_list_orig=os.listdir(TRAIN_IMAGE_PATH)
train_mask_list_orig=os.listdir(TRAIN_MASK_PATH)

train_image_list=[TRAIN_IMAGE_PATH+i for i in train_image_list_orig]
train_mask_list=[TRAIN_MASK_PATH+i for i in train_mask_list_orig]

valid_image_list_orig=os.listdir(VALID_IMAGE_PATH)
valid_mask_list_orig=os.listdir(VALID_MASK_PATH)

valid_image_list=[VALID_IMAGE_PATH+i for i in valid_image_list_orig]
valid_mask_list=[VALID_MASK_PATH+i for i in valid_mask_list_orig]
img=cv2.imread(valid_image_list[50])
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

mask=cv2.imread(valid_mask_list[50])
mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Sample Image")

plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Sample Mask")


print("Image Shape: ",img.shape)
print("Mask Shape: ",mask.shape,"\n\n")


train_image_filenames=tf.constant(train_image_list)
train_mask_filenames=tf.constant(train_mask_list)

valid_image_filenames=tf.constant(valid_image_list)
valid_mask_filenames=tf.constant(valid_mask_list)

train_dataset=tf.data.Dataset.from_tensor_slices((train_image_filenames,train_mask_filenames))
valid_dataset=tf.data.Dataset.from_tensor_slices((valid_image_filenames,valid_mask_filenames))

for image,mask in valid_dataset.take(4):
    print(image)
    print(mask)
    print()

mask=tf.io.read_file(train_mask_list[0])
mask=tf.image.decode_png(mask,channels=3)

class_mask=tf.zeros([mask.shape[0],mask.shape[1]])

for color,class_id in COLORS.items():
    boolean=tf.reduce_all(mask==color,axis=-1)
    class_mask=tf.where(boolean,class_id,class_mask)

class_mask=tf.cast(class_mask,tf.int32)
class_mask

train_image_ds=train_dataset.map(utils.process_path)
processed_train_ds=train_image_ds.map(utils.preprocess)

valid_image_ds=valid_dataset.map(utils.process_path)
processed_valid_datset=valid_image_ds.map(utils.preprocess)


print(train_image_ds)
print()
print(processed_valid_datset)


unet_model=model_utils.unet((96,128,3))
unet_model.summary()

unet_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"]
)

print(processed_train_ds)
print()
print(processed_valid_datset)

print("\n\n")
EPOCHS=100
BUFFER_SIZE=500
BATCH_SIZE=32

train_dataset=processed_train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_dataset=processed_valid_datset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model_history=unet_model.fit(train_dataset,epochs=EPOCHS,validation_data=valid_dataset)

plt.title("Training And Validation Accuracy")
plt.plot(model_history.history["accuracy"],label="train")
plt.plot(model_history.history['val_accuracy'],label="val")
plt.legend()

unet_model.evaluate(valid_dataset)