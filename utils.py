import tensorflow as tf
import pandas as pd

COLORS=pd.read_csv("C:/Users/PratushPc/OneDrive/Documents/COURSES/DL SPECIALIZATION/4. Convolutional Neural Networks/3. Object Detection - Image Segmentation/Lab/Practice_Image/CamVid/class_dict.csv")

COLORS=COLORS[["r","g","b"]].to_numpy()

COLORS={tuple(value):key for key,value in enumerate(COLORS)}
print(len(COLORS))

def rgb_to_class(mask):
    
    class_mask=tf.zeros([mask.shape[0],mask.shape[1]],dtype=tf.int32)

    for color,class_id in COLORS.items():
        boolean=tf.reduce_all(mask==color,axis=-1)
        class_mask=tf.where(boolean,class_id,class_mask)

    class_mask=tf.cast(class_mask,tf.int32)
    return class_mask

def process_path(img_path,mask_path):
    img=tf.io.read_file(img_path)
    img=tf.image.decode_png(img,channels=3)
    img=tf.image.convert_image_dtype(img,tf.float32)

    mask=tf.io.read_file(mask_path)
    mask=tf.image.decode_png(mask,channels=3)
    
    return img,mask

def preprocess(img,mask):
    final_img=tf.image.resize(img,[96,128],method="nearest")
    final_mask=tf.image.resize(mask,[96,128],method="nearest")
    final_mask=rgb_to_class(final_mask)

    return final_img,final_mask