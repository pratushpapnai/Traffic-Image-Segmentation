import tensorflow as tf
import tensorflow.keras.layers as tfl

def conv_block(input=None,n_filters=32,dropout_prob=0,max_pool=True):

    conv=tfl.Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal")(input)
    conv=tfl.Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal")(conv)

    if(dropout_prob>0):
        conv=tfl.Dropout(dropout_prob)(conv)

    if max_pool:
        next_layer=tfl.MaxPool2D((2,2))(conv)

    else:
        next_layer=conv

    skip_connection=conv

    return next_layer,skip_connection



def up_block(expansive_input,contractive_input,n_filters=64):

    up=tfl.Conv2DTranspose(n_filters,(3,3),padding="same",strides=2)(expansive_input)

    merge=tfl.concatenate([up,contractive_input],axis=3)

    conv=tfl.Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal")(merge)
    conv=tfl.Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal")(conv)

    return conv



def unet(input_size,n_filters=64,n_classes=32):

    input=tfl.Input(input_size)

    cblock1=conv_block(input,n_filters)
    cblock2=conv_block(cblock1[0],n_filters*2)
    cblock3=conv_block(cblock2[0],n_filters*4)
    cblock4=conv_block(cblock3[0],n_filters*8,dropout_prob=0.3)

    bottleneck=conv_block(cblock4[0],n_filters*16,dropout_prob=0.3,max_pool=False)

    ublock4=up_block(bottleneck[0],cblock4[1],n_filters*8)
    ublock3=up_block(ublock4,cblock3[1],n_filters*4)
    ublock2=up_block(ublock3,cblock2[1],n_filters*2)
    ublock1=up_block(ublock2,cblock1[1],n_filters)

    conv=tfl.Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal")(ublock1)

    output=tfl.Conv2D(n_classes,1,padding="same")(conv)

    model=tf.keras.models.Model(inputs=input,outputs=output)

    return model