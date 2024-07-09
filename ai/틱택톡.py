import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#1 데이터 읽어오는 함수
# csv파일을 txt파일로 불러온다 txt파일로 불러 오면 ", " 로 구분되어 나누어진다.
# 첫번째 행은 헤더값이 온다. 따라서 skiprows = 1 을 통해 1행을 무시하고 넘어간다.
# converters 를 통해서 문자열을 정수형으로 바꾼다.
def load_Iris(shuffle=False):   
    label={'true':0, 'false':1, 'x': 2, 'b': 3, 'o': 4}

    data = np.loadtxt(r'C:\Users\dltjd\Artificial Intelligence\tic-tac-toe.csv', skiprows=1, delimiter=',',
                      converters={i: lambda name: label[name.decode()] for i in range(10)})
    #셔플이 트루라면 데이터를 섞는다.
    if shuffle:
        np.random.shuffle(data)
    return data

# 데이터에서 훈련세트와 테스트세트를 나눈다.
# test_rate -> 테스트 세트 40% iris_data -> 훈련세트 60%
# 80%를 학습하고 20%는 성능을 테스
def train_test_data_set(iris_data, test_rate=0.2): # train: 0.8, test: 0.2
    n = int(iris_data.shape[0]*(1-test_rate)) # iris_data -> (958, 10) 중 [0] 즉, 958 * 0.8 (766)개 훈련데이터로 설
    #훈련 데이터 574개 
    x_train = iris_data[:n,:-1]  # 958의 행중 1행부터 765행까지 사용 + 마지막 열 제거 (입력)
    y_train = iris_data[:n, -1]  # 958의 행중 1행부터 765행까지 사용 + 마지막 열만 사용  (출력)
    # 테스트 데이터 384
    x_test = iris_data[n:,:-1]  # 766행 부터 958행 까지 사용 + 마지막 열 제거 (입력)
    y_test = iris_data[n:,-1]   # 766행 부터 958행 까지 사용 + 마지막 열만 사용 (출력)
    return (x_train, y_train), (x_test, y_test)

# 데이터 읽어오는 함수를 사용하여 데이터를 불러온다
# 데이터를 섞어서 불러온다 (shuffle = true) -> 데이터를 섞어야 학습이 더 잘된다.
iris_data = load_Iris(shuffle=True)
(x_train, y_train), (x_test, y_test) = train_test_data_set(iris_data, test_rate=0.4) #test데이터의 비율을 40%로 높인다.
print("x_train.shape:", x_train.shape) #(574,4)
print("y_train.shape:", y_train.shape) #(574, 1)
print("x_test.shape:",  x_test.shape)  #(384, 4)
print("y_test.shape:",  y_test.shape)  #(384, 1)
  
# one-hot encoding: 'mse', 'categorical_crossentropy'
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
##print("y_train=", y_train)
##print("y_test=", y_test)

#2
# 모델을 만든다. n은 뉴런의 갯수 , input_dim은 입력갯수, 첫벗째 input층 마지막줄 output층
# output층은 3가지 종류로 인식할것이기 때문에 unit = 3 
n = 30  # number of neurons in a hidden layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=n, input_dim=9, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.summary()

#3
def MSE(y, t):
    return tf.reduce_mean(tf.square(y - t)) # (y - t)**2

CCE = tf.keras.losses.CategoricalCrossentropy()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
##model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
##model.compile(optimizer=opt, loss= MSE, metrics=['accuracy'])
##model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# model.compile을 통해서 학습환경을 정한다. metrics -> 정확도를 출력 
model.compile(optimizer=opt, loss= CCE, metrics=['accuracy'])

# model.fit을 통해 학습하고 값을 ret에 저장
ret = model.fit(x_train, y_train, epochs=1000, batch_size=7, verbose=2) # batch_size=32
print("len(model.layers):", len(model.layers))  # 2
loss = ret.history['loss']
plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#4
##print(model.get_weights())
##for i in range(len(model.layers)):
##    print("layer :", i, '-'*20)
##    w = model.layers[i].weights[0].numpy()
##    b = model.layers[i].bias.numpy()
##    print("weights[{}]: {}".format(i, np.array2string(w)))
##    print("bias[{}]:    {}".format(i, np.array2string(b)))

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
print("confusion_matrix(C):", C)
