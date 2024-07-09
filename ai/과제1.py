import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 데이터에 csv파일 데이터를 불러온다. encoding을 통해 한글을 불러올수있게한다. 칼럼의 헤더는 7번라인
data = pd.read_csv(r'C:\Users\dltjd\Artificial Intelligence\month_temp.csv', encoding = 'cp949', header=6)

# csv파일의 년월 중 년도는 필요하지 않고 월만 필요하기 때문에 월의 값만 가지고온다.
x_month = data['년월'].str.split("-").str[1].astype(float).values 

# csv 파일에서 월별 평균기온을 가져온다. npArray로 가져옴.
y_temp = data['평균기온(℃)'].values

# 정규화
#x_month /= max(x_month)
#y_temp /= max(y_temp)

# n-차 다항식 회귀
n = 3

# x의 데이터갯수 (20개) x n + 1개 (4개)의 배열을 만들고 값을 1로 초기화
X = np.ones(shape = (len(x_month), n+1), dtype=np.float32)

#0번째 열을 1로 초기화하고 배열 X에서 1번째 열부터 n번째 열까지 x^i 값을 차례로 넣어준다. 
for i in range(1, n+1):
     X[:, i] = x_month**i

#신경망 입력 정의 n+1은 입력 데이터의 형태를 나타냄.
inputs = tf.keras.layers.Input(shape=(n+1,))
# 신경망의 출력을 정의 units = 1은 뉴런 1개 bias는 없음을 뜻함 bias는 없어도 될수있다.
# 왜냐하면, 상수항이 X배열의 1열에 1이 초기화 되어 있기때문이다.
# (inputs) 이 Dense 층의 입력은 앞에서 정의한 input 사용하겠다는 
outputs = tf.keras.layers.Dense(units=1, use_bias=False)(inputs)
# 실제 신경망을 만든다.
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 신경망을 출력 
model.summary()

 
#rmsprop은 옵티마이저의 알고리즘의 하나  학습률은 0.1 을 opt에 저장
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.1)
#옵티마이저의 알고리즘하나인 rmsprop을 저장한 opt를 옵티마이저를 사용한다.
#그리고 mse는 sum((트루값 - 예측값)^2) / t.size()
model.compile(optimizer=opt, loss='mse')

#모델을 학습하는 함수 fit() X는 입력데이터 , y_month는 정답 , 10000번 반복 ,
#batch_size를 통해 x의 값을 쪼갠다 10000번의 포문안에 6번의 포문이 계속해서 돈다. 
#verbose = 2 한줄씩 출력 0은 결과만 출력 
ret = model.fit(X, y_temp, epochs = 4000, batch_size=2, verbose = 2)




#1: 모델 전체 저장
import os
if not os.path.exists("./RES1000"):
     os.mkdir("./RES1000")
model.save("./RES1000/1401.keras")   # HDF5, keras format

#2: 모델 구조 저장
json_string = model.to_json()
import json
file = open("./RES1000/1401.model", 'w')
json.dump(json_string, file)
file.close()
 
#3: 가중치 저장
model.save_weights("./RES1000/weights/1401")
 
#4: 학습중에 체크포인트 저장
filepath = "RES1000/ckpt/1401-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath, verbose=0, save_weights_only=True, save_freq=50)
ret = model.fit(X, y_temp, epochs=100, callbacks = [cp_callback], verbose=0)





print("len(model.layers):", len(model.layers)) # 2

loss = ret.history['loss']
print("loss:", loss[-1])
#print(model.get_weights())  # weights
print("weights:", model.layers[1].weights[0].numpy())

plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.scatter(x_month, y_temp) 
y_pred = model.predict(X)
plt.plot(x_month, y_pred, color='red')
plt.show()

