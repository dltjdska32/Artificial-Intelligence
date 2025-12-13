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
  <img src="data:image/png;base64,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" alt="TensorFlow" height="50">
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



