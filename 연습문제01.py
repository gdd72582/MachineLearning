# -*- coding: utf-8 -*-
"""혼공머신.ipynb

# 01-3 ~ 02-2 연습 문제

## 1번 문제
문제 1: K-최근접 이웃 (KNN) 분류 문제

다음은 특정 물고기의 길이와 무게 데이터입니다:
	•	길이: [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
	•	무게: [5.0, 10.0, 15.0, 20.0, 30.0, 35.0, 40.0]

이 데이터로 KNN 분류기를 훈련하고, 길이가 18.0이고 무게가 25.0인 물고기를 분류했을 때 결과를 예측하세요. K=3으로 설정하세요.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 데이터 생성
length = [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
weight = [5.0, 10.0, 15.0, 20.0, 30.0, 35.0, 40.0]
fish_data = np.column_stack((length, weight))
fish_target = np.concatenate((np.ones(3), np.zeros(4)))

# 데이터 분할 (훈련 세트와 테스트 세트)
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42
)

# K-최근접 이웃 모델 생성 및 훈련
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_input, train_target)

# 모델 평가
print("테스트 세트 정확도:", kn.score(test_input, test_target))

# 새로운 데이터 예측
new_fish = [[18.0, 25.0]]
predicted_label = kn.predict(new_fish)
print("새로운 데이터 예측 결과:", predicted_label)

# 새로운 데이터가 속한 클래스 확인
if predicted_label[0] == 1:
    print("예측: Class 1 (도미)")
else:
    print("예측: Class 0 (빙어)")

"""## 2번 문제
문제 2: 데이터 전처리 및 표준화

주어진 물고기 데이터의 길이와 무게를 각각 표준화하려고 합니다. 데이터는 다음과 같습니다:
	•	길이: [30, 35, 40, 45, 50]
	•	무게: [300, 400, 500, 600, 700]

	1.	주어진 데이터를 표준화하세요 (평균과 표준편차를 사용).
	2.	표준화된 데이터로 (길이=38, 무게=450)인 데이터를 새로운 데이터셋과 비교했을 때, 가장 가까운 데이터 포인트의 인덱스를 찾으세요.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 데이터
length = [30, 35, 40, 45, 50]
weight = [300, 400, 500, 600, 700]

fish_data = np.column_stack((length, weight))

# 각 열에 대해 평균과 표준편차를 계산
mean = np.mean(fish_data, axis=0)
std = np.std(fish_data, axis=0)

# 표준화
fish_scaled_data = (fish_data - mean) / std
print(fish_scaled_data)

# 새로운 데이터 표준화
new = ([38, 450] - mean) / std

# KNN 설정 및 가장 가까운 점 계산
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(fish_scaled_data, [0, 1, 2, 3, 4])  # 데이터 인덱스를 타겟으로 사용
distance, index = kn.kneighbors([new])

# 시각화
plt.scatter(fish_scaled_data[:, 0], fish_scaled_data[:, 1])
plt.scatter(new[0], new[1], marker='^', label="New Point")
plt.scatter(fish_scaled_data[index, 0], fish_scaled_data[index, 1], marker='D', label="Closest Point")
plt.xlabel('Length (scaled)')
plt.ylabel('Weight (scaled)')
plt.legend()
plt.show()
