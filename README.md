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
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="TensorFlow" height="80" style="margin: 10px;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" alt="Keras" height="80" style="margin: 10px;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy" height="80" style="margin: 10px;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" alt="Pandas" height="80" style="margin: 10px;">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" alt="Matplotlib" height="80" style="margin: 10px;">
</div>




