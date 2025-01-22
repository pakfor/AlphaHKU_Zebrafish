# Import modules for data manipulation
import numpy as np
import os  # For Windows file system manipulations
import time
import random
import cv2  # For general image I/O
import skimage  # For histogram equalization
import albumentations  # For data augmentation
import matplotlib.pyplot as plt  # For visualization

# Import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision  # For mixed precision computing

random.seed(0)

os.chdir("G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation")

# Enable FP16 mixed precision training
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

ENABLE_DATA_AUG = True

#%% Data Augmentation

def data_augmentation(x_train, y_train):
    trans_1 = albumentations.Compose([
        albumentations.HorizontalFlip(p=1.0)])
    trans_2 = albumentations.Compose([
        albumentations.VerticalFlip(p=1.0)])
    trans_3 = albumentations.Compose([
        albumentations.HorizontalFlip(p=1.0),
        albumentations.VerticalFlip(p=1.0)])
    return [trans_1(image=x_train)['image'], trans_2(image=x_train)['image'], trans_3(image=x_train)['image']], \
           [trans_1(image=y_train)['image'], trans_2(image=y_train)['image'], trans_3(image=y_train)['image']]

# FOR DEBUG
# img = cv2.imread("G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/image/zbf_4_img_1020_png_Box_05.png", cv2.IMREAD_GRAYSCALE)
# img = skimage.exposure.equalize_hist(img)
# mask = np.load("G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/label/zbf_4_img_1020_png_Box_05.npy")

# img_trans, mask_trans = data_augmentation(x_train=img, y_train=mask)

# img_trans_1 = img_trans[0]; img_trans_2 = img_trans[1]; img_trans_3 = img_trans[2]
# mask_trans_1 = mask_trans[0]; mask_trans_2 = mask_trans[1]; mask_trans_3 = mask_trans[2]

# plt.imshow(img_trans_1, cmap='gray');  plt.show()
# plt.imshow(mask_trans_1, cmap='gray'); plt.show()
# plt.imshow(img_trans_2, cmap='gray');  plt.show()
# plt.imshow(mask_trans_2, cmap='gray'); plt.show()
# plt.imshow(img_trans_3, cmap='gray');  plt.show()
# plt.imshow(mask_trans_3, cmap='gray'); plt.show()


#%% Data Import

label_dir = "./Data/label"
img_dir = "./Data/image"
label_list = os.listdir(label_dir)
random.shuffle(label_list)

train_test_split = 0.7
data_size = len(label_list)
train_size = int(data_size * train_test_split)
test_size = data_size - train_size

train_label_list = label_list[0:train_size]
test_label_list = label_list[train_size:]
assert len(train_label_list) == train_size
assert len(test_label_list) == test_size

x_train = None
y_train = None
for lb_dir in train_label_list:
    img_tmp = cv2.imread(f"{img_dir}/{lb_dir.replace('npy','png')}", cv2.IMREAD_GRAYSCALE)
    img_tmp = skimage.exposure.equalize_hist(img_tmp)
    lb_tmp = np.load(f"{label_dir}/{lb_dir}")

    # Data Augmentation
    if ENABLE_DATA_AUG:
        img_trans, mask_trans = data_augmentation(x_train=img_tmp, y_train=lb_tmp)
        img_tmp_l = [img_tmp] + img_trans
        lb_tmp_l = [lb_tmp] + mask_trans
    else:
        img_tmp_l = [img_tmp]
        lb_tmp_l = [lb_tmp]

    for img_tmp, lb_tmp in zip(img_tmp_l, lb_tmp_l):
        img_tmp = np.expand_dims(img_tmp, axis=0)
        img_tmp = np.expand_dims(img_tmp, axis=-1)
        img_tmp = img_tmp.astype('float32')
        lb_tmp = np.expand_dims(lb_tmp, axis=0)
        lb_tmp = lb_tmp.astype('float32')
        if x_train is None:
            x_train = img_tmp
        else:
            x_train = np.append(x_train, img_tmp, axis=0)
        if y_train is None:
            y_train = lb_tmp
        else:
            y_train = np.append(y_train, lb_tmp, axis=0)

x_test = None
y_test = None
for lb_dir in test_label_list:
    img_tmp = cv2.imread(f"{img_dir}/{lb_dir.replace('npy','png')}", cv2.IMREAD_GRAYSCALE)
    img_tmp = skimage.exposure.equalize_hist(img_tmp)
    lb_tmp = np.load(f"{label_dir}/{lb_dir}")

    # Data Augmentation (May not need for test set!)
    if ENABLE_DATA_AUG:
        img_trans, mask_trans = data_augmentation(x_train=img_tmp, y_train=lb_tmp)
        img_tmp_l = [img_tmp] + img_trans
        lb_tmp_l = [lb_tmp] + mask_trans
    else:
        img_tmp_l = [img_tmp]
        lb_tmp_l = [lb_tmp]

    for img_tmp, lb_tmp in zip(img_tmp_l, lb_tmp_l):
        img_tmp = np.expand_dims(img_tmp, axis=0)
        img_tmp = np.expand_dims(img_tmp, axis=-1)
        img_tmp = img_tmp.astype('float32')
        lb_tmp = np.expand_dims(lb_tmp, axis=0)
        lb_tmp = lb_tmp.astype('float32')
        if x_test is None:
            x_test = img_tmp
        else:
            x_test = np.append(x_test, img_tmp, axis=0)
        if y_test is None:
            y_test = lb_tmp
        else:
            y_test = np.append(y_test, lb_tmp, axis=0)

print("Train Set Shape:", x_train.shape, y_train.shape)
print("Test Set Shape:", x_test.shape, y_test.shape)

#x_train = x_train / 255.0
#x_test = x_test / 255.0


# Shuffle the data
shuffler = np.random.permutation(len(x_train))
x_train = x_train[shuffler]
y_train = y_train[shuffler]

shuffler = np.random.permutation(len(x_test))
x_test = x_test[shuffler]
y_test = y_test[shuffler]

#%% Data Generator

batch_size = 4

# Define generator for training data according to batch size as defined by user
def train_gen(x_train,y_train):
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    while True:
        for i in range(0,len(x_train),batch_size):
            yield (x_train[i:i+batch_size],y_train[i:i+batch_size])

# Define generator for testing data, batch size is set to 1
def validate_gen(x_test,y_test):
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]
    
    while True:
        for i in range(0,len(x_test),1):
            yield (x_test[i:i+1],y_test[i:i+1])
    
#%% Neural Network

# Define the architecture U-Net of the neural network to be trained for segmentation
def UNet():
    input = layers.Input(shape=(128,128,1))
    x = layers.Conv2D(64,3,1,padding='same')(input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a1 = x
    
    x = layers.MaxPooling2D(2,2)(a1)
    
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a2 = x
    
    x = layers.MaxPooling2D(2,2)(a2)
    
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a3 = x
    
    x = layers.MaxPooling2D(2,2)(a3)
    
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a4 = x
    
    x = layers.MaxPooling2D(2,2)(a4)
    
    x = layers.Conv2D(1024,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1024,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a4],axis=-1)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a3],axis=-1)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a2],axis=-1)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(16,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a1],axis=-1)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(1,1,1,padding='same',activation='sigmoid')(x)
    
    model = tf.keras.Model(input,x)
    model.summary()
    
    return model

#%% Model Compile & Training

# Start the timer for recording the time taken for training the model
train_start = time.time()

model_unique_name = round(train_start)

# Create saving paths
save_dir = f'./Model/{model_unique_name}'
os.makedirs(save_dir, exist_ok=True)

model_save = f'{save_dir}/model' # For saving the result at the end of the training
log_save = f'{save_dir}/log' # For storing training activities
ckpt_save = f'{save_dir}/model_ckpt' # For saving the best weightings recorded during the training (continuously updated during training)

# Create the directories as mentioned above
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists(log_save):
    os.mkdir(log_save)
if not os.path.exists(ckpt_save):
    os.mkdir(ckpt_save)

# Define optimizer: Adam
opt_adam = tf.keras.optimizers.Adam()

# Define callbacks: early stopping and model checkpoint
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 20, verbose = 1, mode = 'min')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')

# Make model
model = UNet()

# Compile the model by defining loss function (MSE) and optimizer (Adam)
model.compile(opt_adam,loss='mean_squared_error',metrics=['accuracy'])

# Fit the model with training data and validate it with testing data every epoch
# Define maximum number of epochs
# Apply callbacks
model.fit(train_gen(x_train,y_train), steps_per_epoch = len(x_train)/batch_size, epochs = 500, validation_data = validate_gen(x_test,y_test), validation_steps = len(x_test), callbacks = [model_ckpt])

# Stop the timer for recording the time taken for training the model
train_end = time.time()
# Calculate time taken and print it
elapse = train_end - train_start
print(elapse)

# Save the trained model to the designated directory
model.save(model_save)