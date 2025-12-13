# 인공지능 과제

## 프로젝트 구조

```
Artificial-Intelligence/
├── ai/
│   ├── mid_assignment/          # 중간 과제
│   │   ├── month_temp_ml.py     # 월별 온도 예측 (다항식 회귀)
│   │   └── month_temp.csv        # 온도 데이터
│   │
│   └── final_assignment/         # 최종 과제
│       ├── cloth_ml/             # 의류 이미지 분류
│       │   ├── finalAssignment.py
│       │   └── clothData/        # 학습/테스트 이미지 데이터
│       │
│       └── tic_tac_toc_ml/       # 틱택토 게임 분류
│           ├── 틱택톡.py
│           └── tic-tac-toe.csv
```

## 중간 과제

### 월별 온도 예측
- **파일**: `ai/mid_assignment/month_temp_ml.py`
- **목적**: 월별 평균 기온 예측
- **방법**: 3차 다항식 회귀 모델

## 최종 과제

### 1. 의류 이미지 분류
- **파일**: `ai/final_assignment/cloth_ml/finalAssignment.py`
- **목적**: 의류 색상 분류 (black, gray, white)
- **방법**: VGG16 전이학습 (Transfer Learning)
- **데이터**: train/test 이미지 데이터셋

### 2. 틱택토 게임 분류
- **파일**: `ai/final_assignment/tic_tac_toc_ml/틱택톡.py`
- **목적**: 틱택토 게임 상태 분류
- **방법**: 신경망 분류 모델

## 사용 기술

<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png" alt="TensorFlow" height="50">
</div>

<div align="center">
  <img src="https://keras.io/img/logo.png" alt="Keras" height="50">
</div>

<div align="center">
  <img src="https://numpy.org/images/logo.svg" alt="NumPy" height="50">
</div>

<div align="center">
  <img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="Pandas" height="50" style="background-color: white; padding: 5px;">
</div>

<div align="center">
  <img src="https://matplotlib.org/stable/_static/logo2_compressed.svg" alt="Matplotlib" height="50">
</div>



