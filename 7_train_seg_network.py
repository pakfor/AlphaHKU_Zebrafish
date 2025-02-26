# Import modules for data manipulation
import numpy as np
import os  # For Windows file system manipulations
import time
import random
import cv2  # For general image I/O
import skimage  # For histogram equalization
import albumentations  # For data augmentation
import argparse  # For parsing arguments

# Import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K  # For custom loss

random.seed(0)

# os.chdir("G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation")

# Enable FP16 mixed precision training
# from tensorflow.keras import mixed_precision  # For mixed precision computing
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


# Data Augmentation
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


# DICE
def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# MSE
def mean_squared_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)


# MSE + DICE
def total_loss_mse_dice_mixed(y_true, y_pred):
    return (dice_coef_loss(y_true, y_pred) + mean_squared_loss(y_true, y_pred)) / 2


def total_loss(y_true, y_pred, loss_method):
    if loss_method == "mse":
        return mean_squared_loss(y_true, y_pred)
    elif loss_method == "dice":
        return dice_coef_loss(y_true, y_pred)
    elif loss_method == "mse_dice_mixed":
        return total_loss_mse_dice_mixed(y_true, y_pred)
    else:
        return None


# Calculate the gradient
def grad(model, img, mask, loss_method, training=True):
    with tf.GradientTape() as tape:
        y_pred = model(img, training)
        loss = total_loss(mask, y_pred, loss_method)
        return loss, tape.gradient(loss, model.trainable_variables)


# Save best model
def save_best_model(model, save_dir, epoch, val_loss_avg):
    val_loss_avg = str(round(val_loss_avg, 4))
    epoch = str(epoch)
    model_save_dir = f"{save_dir}/epoch_{epoch}_loss_{val_loss_avg}"
    model.save(model_save_dir)
    return model_save_dir


def main(x_train_path, y_train_path, preprocess_method, hist_ref_path, enable_data_aug, model_save_path, train_test_split, loss_method, num_epoch, batch_size):
    img_dir = x_train_path  # G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/image
    label_dir = y_train_path  # G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/label
    preprocess_method = preprocess_method  # hist_eq/hist_match
    hist_ref_path = hist_ref_path  # G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/image/zbf_4_img_0990_png_Box_04.png
    enable_data_aug = enable_data_aug  # True
    model_save_path = model_save_path  # G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Model
    train_test_split = train_test_split # 0.7
    loss_method = loss_method  # mse_dice_mixed
    num_epochs = num_epoch # 500
    batch_size = batch_size # 4

    # Data size
    label_list = os.listdir(label_dir)
    random.shuffle(label_list)
    data_size = len(label_list)
    train_size = int(data_size * train_test_split)
    test_size = data_size - train_size
    train_label_list = label_list[0:train_size]
    test_label_list = label_list[train_size:]
    assert len(train_label_list) == train_size
    assert len(test_label_list) == test_size
    
    # Load and split data
    x_train = None
    y_train = None
    for lb_dir in train_label_list:
        img_tmp = cv2.imread(f"{img_dir}/{lb_dir.replace('npy','png')}", cv2.IMREAD_GRAYSCALE)
        if preprocess_method == "hist_eq":
            img_tmp = skimage.exposure.equalize_hist(img_tmp)
        elif preprocess_method == "hist_match":
            ref_img = cv2.imread(hist_ref_path, cv2.IMREAD_GRAYSCALE)
            img_tmp = np.expand_dims(img_tmp, axis=-1)
            ref_img = np.expand_dims(ref_img, axis=-1)
            img_tmp = skimage.exposure.match_histograms(img_tmp, ref_img, channel_axis=-1)
            img_tmp = np.squeeze(img_tmp, axis=-1)
            img_tmp = (img_tmp - np.min(img_tmp)) / (np.max(img_tmp) - np.min(img_tmp))
        else:
            img_tmp = (img_tmp - np.min(img_tmp)) / (np.max(img_tmp) - np.min(img_tmp))
        lb_tmp = np.load(f"{label_dir}/{lb_dir}")

        # Data Augmentation
        if enable_data_aug:
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
        if preprocess_method == "hist_eq":
            img_tmp = skimage.exposure.equalize_hist(img_tmp)
        elif preprocess_method == "hist_match":
            ref_img = cv2.imread(hist_ref_path, cv2.IMREAD_GRAYSCALE)
            img_tmp = np.expand_dims(img_tmp, axis=-1)
            ref_img = np.expand_dims(ref_img, axis=-1)
            img_tmp = skimage.exposure.match_histograms(img_tmp, ref_img, channel_axis=-1)
            img_tmp = np.squeeze(img_tmp, axis=-1)
            img_tmp = (img_tmp - np.min(img_tmp)) / (np.max(img_tmp) - np.min(img_tmp))
        else:
            img_tmp = (img_tmp - np.min(img_tmp)) / (np.max(img_tmp) - np.min(img_tmp))
        lb_tmp = np.load(f"{label_dir}/{lb_dir}")

        # Data Augmentation (May not need for test set!)
        if enable_data_aug:
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
    
    # Shuffle the data
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]
    
    # Define model architecture
    model = UNet()
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    train_start = time.time()
    # Define directories for storing model weights
    model_unique_name = round(train_start)
    save_dir = f'{model_save_path}/{model_unique_name}'
    os.makedirs(save_dir, exist_ok=True)
    ckpt_save = f'{save_dir}/model_ckpt'
    os.makedirs(ckpt_save, exist_ok=True)
    
    # Set up for training loop
    best_val_loss = 1.0
    train_steps_per_epoch = int(len(x_train) / batch_size) + 1
    test_steps_per_epoch = int(len(x_test) / batch_size) + 1

    # Training loop
    for epoch in range(0, num_epochs):
        epoch_start = time.time()
        train_loss_in_epoch = []
        val_loss_in_epoch = []
    
        print('Epoch {} :'.format(epoch))
        train_set = train_gen(x_train, y_train)
        test_set = validate_gen(x_test, y_test)
    
        # Train
        print("Training ", end = '')
        print("  | ", end='')
        for batch_idx in range(0, train_steps_per_epoch):
            x_train_in_batch, y_train_in_batch = next(train_set)
            train_loss_in_batch, train_grad_in_batch = grad(model, x_train_in_batch, y_train_in_batch, loss_method, True)
            optimizer.apply_gradients(zip(train_grad_in_batch, model.trainable_variables))
            train_loss_in_epoch.append(train_loss_in_batch)
            print('=', end='')
        print(" |")
        print("")
    
        # Validation
        print("Validation ", end = '')
        print("  | ", end='')
        for batch_idx in range(0, test_steps_per_epoch):
            x_test_in_batch, y_test_in_batch = next(test_set)
            test_loss_in_batch, _ = grad(model, x_test_in_batch, y_test_in_batch, loss_method, False)
            val_loss_in_epoch.append(test_loss_in_batch)
            print('=', end='')
        print(" |")
        print("")
    
        current_epoch_avg_val_loss = np.mean(val_loss_in_epoch)
        current_epoch_avg_train_loss = np.mean(train_loss_in_epoch)
        # print(f"Average Validation Loss: {current_epoch_avg_val_loss}")
    
        epoch_end = time.time()
        epoch_time_spent = str(round(epoch_end - epoch_start))
        total_time_spent = str(round(epoch_end - train_start))
        print(' Train Loss = {:.4f}, Validation Loss = {:.4f}'.format(current_epoch_avg_train_loss, current_epoch_avg_val_loss))
        print(f" Elapsed time for epoch {epoch}: {epoch_time_spent} seconds")
        print(f" Total elapsed time for training: {total_time_spent} seconds")
    
        # Save best weights
        if current_epoch_avg_val_loss < best_val_loss:
            print("Best weights found, saving...")
            model_save_dir = save_best_model(model, ckpt_save, epoch, current_epoch_avg_val_loss)
            best_val_loss = current_epoch_avg_val_loss
            print(f"Best weights saved to {model_save_dir}")
        print(f"Lowest validation loss so far: {best_val_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_train", help="Full path to the folder where the input data is stored", type=str)
    parser.add_argument("--y_train", help="Full path to the folder where the output data is stored", type=str)
    parser.add_argument("--preprocess", help="Pre-processing technique", type=str)
    parser.add_argument("--hist_ref", help="Histogram reference image (Only applicable for using preprocessing method hist_match", type=str)
    parser.add_argument("--enable_data_aug", help="Enable data augmentation", action="store_true")
    parser.add_argument("--model_save_path", help="Full path the folder where the trained model will be stored", type=str)
    parser.add_argument("--train_test_split", help="Train-test split ratio", type=float)
    parser.add_argument("--loss", help="Loss function for the model", type=str)
    parser.add_argument("--epoch", help="Epoch", type=int)
    parser.add_argument("--batch_size", help="Batch size", type=int)
    args = parser.parse_args()

    x_train_path = args.x_train
    y_train_path = args.y_train
    preprocess_method = args.preprocess
    hist_ref_path = args.hist_ref
    enable_data_aug = args.enable_data_aug
    model_save_path = args.model_save_path
    train_test_split = args.train_test_split
    loss_method = args.loss
    epoch = args.epoch
    batch_size = args.batch_size

    print("x_train data path", x_train_path)
    print("y_train data path", y_train_path)
    print("Preprocessing", preprocess_method)
    print("Histogram reference image", hist_ref_path)
    print("Data augmentation", enable_data_aug)
    print("Model saving path", model_save_path)
    print("Train-test split", train_test_split)
    print("Loss method", loss_method)
    print("Epoch", epoch)
    print("Batch size", batch_size)

    main(x_train_path, y_train_path, preprocess_method, hist_ref_path, enable_data_aug, model_save_path, train_test_split, loss_method, epoch, batch_size)


# Debug for data augmentation
# img = cv2.imread("G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/image/zbf_4_img_1020_png_Box_05.png", cv2.IMREAD_GRAYSCALE)
# img = skimage.exposure.equalize_hist(img)
# mask = np.load("G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/label/zbf_4_img_1020_png_Box_05.npy")

# img_trans, mask_trans = data_augmentation(x_train=img, y_train=mask)

# img_trans_1 = img_trans[0]; img_trans_2 = img_trans[1]; img_trans_3 = img_trans[2]
# mask_trans_1 = mask_trans[0]; mask_trans_2 = mask_trans[1]; mask_trans_3 = mask_trans[2]

# import matplotlib.pyplot as plt  # For visualization
# plt.imshow(img_trans_1, cmap='gray');  plt.show()
# plt.imshow(mask_trans_1, cmap='gray'); plt.show()
# plt.imshow(img_trans_2, cmap='gray');  plt.show()
# plt.imshow(mask_trans_2, cmap='gray'); plt.show()
# plt.imshow(img_trans_3, cmap='gray');  plt.show()
# plt.imshow(mask_trans_3, cmap='gray'); plt.show()