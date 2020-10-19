# Data_mining_for_business
2020년 2학기 비즈니스를 위한 데이터마이닝 수업 실습자료 (이홍주 교수님)    

중간고사 대비
---
- 회귀 문제 : [DAT2-Regression.ipynb](https://github.com/J-TKim/Data_mining_for_business/blob/master/DAT2-Regression.ipynb)
  - 선형 회귀분석 sm.OLS
  - 분산팽창지수 VIF 계산
  - 독립변수 제거 방안 (p-value를 계산한 후진제거법)
  - 교차검증 (K-fold cross validation)
  
  
 - 로지스틱 회귀 : [DAT2-Logistic.ipynb](https://github.com/J-TKim/Data_mining_for_business/blob/master/DAT2-Logistic.ipynb)
   - LogisticRegression
   - 정밀도와 재현율 : precision_recall_curve
   - ROC 곡선 : roc_curve
   
   
  - 릿지, 랏소, 엘라스틱 회귀 : [DAT2-Ridge.ipynb](https://github.com/J-TKim/Data_mining_for_business/blob/master/DAT2-Ridge.ipynb)
    - 릿지 회귀 : sklearn.linear_model.Ridge
    - 라쏘 회귀 : sklearn.linear_model.Lasso
    - 엘라스틱 회귀 : sklearn.linear_model.ElasticNet
    - StandardScaler : 평균이 0이고 표준편차가 1인 표준정규분포를 따르게 변환해줌
    - MinMaxScaler : 최대, 최솟값을 이용해서 0과 1 사이의 범위로 변환
    - GridSearchCV : 그리드 탐색은 하이퍼 파라미터의 범위를 입력해 주고 범위에 해당하는 여러 조합에 대해 학습한 후에 성과를 측정하여 최적의 하이퍼 파라미터를 찾아주는 방안
    - RandomizedSearchCV : 랜덤 탐색은 하이퍼 파라미터로 입력된 값들의 모든 조합을 탐색하지 않고 조합을 무작위로 선정하여 학습하고 성과를 측정함
    - LogisticRegression : 다른 선형 모형처럼 로지스틱 회귀 모형도 L1, L2 노름을 사용하여 규제할 수 있음
      - sklearn 은 LogisticRegression 에서 L2 노름을 기본 규제로 사용하고 있음
      - C 가 규제 정도를 조절하는 파라미터이고, C = 1/alpha 이기에 C 값이 높을 수록 모형의 규제가 줄어듦
      - C 도 하이퍼 파라미터 튜닝을 통해 최적값을 파악하여야 하며, 기본값은 1
