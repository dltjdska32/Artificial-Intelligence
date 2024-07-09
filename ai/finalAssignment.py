import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers   import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
import numpy as np
import matplotlib.pyplot as plt
import os

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2


# 이미지 파일들이 있는 디렉토리 경로(xtrain, xtest)
x_test_images_dir = "C:/Users/dltjd/Artificial Intelligence/clothData/test"
x_train_image_dir = "C:/Users/dltjd/Artificial Intelligence/clothData/train"

# 클래스 레이블 정의
class_labels = {'white': 0, 'gray': 1, 'black': 2}

# 데이터 저장을 위한 리스트 초기화
x_train = []
y_train = []
x_test = []
y_test = []

# 이미지 파일들의 경로를 리스트로 수집
for class_name, class_label in class_labels.items():
    # 훈련 데이터
    train_dir = os.path.join(x_train_image_dir, class_name)
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(train_dir, filename)
            img = load_img(img_path, target_size=(32, 32))
            img_array = img_to_array(img) / 255.0
            x_train.append(img_array)
            y_train.append(class_label)

    # 테스트 데이터
    test_dir = os.path.join(x_test_images_dir, class_name)
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=(32, 32))
            img_array = img_to_array(img) / 255.0
            x_test.append(img_array)
            y_test.append(class_label)

# 리스트를 NumPy 배열로 변환
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.astype('float32') # (사진갯수, 사진높이, 너비, 채널수)
x_test  = x_test.astype('float32')  # (사진갯수, 사진높이, 너비, 채널수)


# one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)


# preprocessing, 'caffe', x_train, x_test: BGR
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#3: resize_layer
inputs = Input(shape=(32, 32, 3))
resize_layer = tf.keras.layers.Lambda( 
                    lambda img: tf.image.resize(img,(224, 224)))(inputs) 

#4:
##W = 'C:/Users/user/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
##W = './Data/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg_model = VGG16(weights='imagenet', include_top= False, 
                     input_tensor= resize_layer) # input_tensor= inputs
vgg_model.trainable=False
##for layer in vgg_model.layers:
##    layer.trainable = False
                      
#4-1: output: classification
x = vgg_model.output
x = Flatten()(x)   # x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outs  = Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outs)
model.summary()

#5: train and evaluate the model
filepath = "RES/ckpt/4404-model.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath, verbose=0, save_best_only=True)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=30, batch_size= 64,
                validation_split=0.2, verbose=0, callbacks = [cp_callback])
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#6: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'],  "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].set_ylim(0, 1.1)
ax[1].plot(ret.history['accuracy'],     "b-", label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label="val accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc='lower right')
fig.tight_layout()
plt.show()

