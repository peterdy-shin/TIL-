#!/usr/bin/env python
# coding: utf-8

# # 탐색적 데이터 분석 : EDA

# ### 개요
# - EDA : Exploratory Data Analysis
# - 이전의 통계학은 적은 표본 데이터로 더 큰 모집단에 대한 결론을 도출하기 위한 과정을 다룸
# - 탐색적 데이터 분석은 통계를 컴퓨터 과학 분야에 접목시키려는 노력에서 탄생됨
# - 1977년 탐색적 데이터 분석이라는 존 투기의 책으로 정립됨
#     - Exploratory Data Analysis. Pearson : Tukey, John W.

# ### 데이터 분석의 절차
# 
# 1. 문제 정의
#     - 분석목적, 가설수립, 비지니스 관계, 분석지표, 분석예상시간등을 수립하는 단계
# 2. 데이터 수집
#     - 문제와 관계가 있는 데이터를 모으는 단계
# 3. 데이터 처리
#     - 모아진 데이터에서 필요한 데이터와 분석하기 쉬운 데이터의 형태로 가공하는 단계
# 4. 데이터 분석
#     - 목적을 달성하기 위해 데이터에서 결과를 얻어내는 단계
#     - 모델링, 알고리즘 생성
#     - 여러가지 분석 방법들을 사용
# 5. 리포팅 및 피드백
#     - 분석된 결과로 상대를 설득
#     
#     
# - 피드백을 통해 분석이 잘못 되었으면 위의 과정 중에 잘못된 과정으로 돌아가 다시 분석
# - EDA의 과정은 데이터 수집 이후 데이터 처리 이전에 수행

# #### 여러가지 분석 방법들
# - 빈도분석(Frequency) : 데이터의 분포를 나타내는 분석 방법 (이산적 변수 : 평균)
# - 기술통계분석(Descriptive) : 데이터의 분포를 나타내는 분석 방법 (연속적 변수 : 시간에 따른 평균)
# - 상관관계분석(Correlation Analysis) : 두 변수의 관계를 나타내는 분석 방법
# - 요인분석(Factor Analysis) : 어떠한 특징을 대표하는 요인을 선별해내는 방법
# - 회귀분석(Regression Analysis) : 데이터의 상관관계를 통해 수치데이터를 예측하는 분석 방법
# - 판별분석(Discriminants Analysis) : 데이터를 통해 결과를 예측해내는 분석 방법 (독립변수로 종속변수를 알아냄)
# - 군집분석(Cluster Analysis) : 종속변수가 없는 데이터를 그룹핑 해주는 분석 방법

# ### 탐색적 데이터 분석의 정의
# - 데이터 분석의 절차에서 데이터 처리 부분에 해당되는 단계
# - 데이터 수집이 완료되고 데이터를 본격적으로 분석하기 전에 수집된 데이터를 다양한 각도에서 관찰하는 과정
# - 그래프나 통계학적인 방법으로 데이터를 직관적으로 바라보는 과정

# ### 탐색적 데이터 분석의 목적
# - 분석을 하기전에 수집하고 처리된 데이터를 파악하고 이해하는것

# #### 1. 정형화된 데이터 요소
# - 정형화 되어 있지 않은 데이터를 정형화된 형태로 변환
# - 연속형 
#     - 일정 범위 내에서 어떤 값이던 사용이 가능한 데이터
#     - 실수형 데이터 (소수점)
#     - 예 : 키, 체중 
# - 이산
#     - 횟수와 같이 정수값을 사용하는 데이터
#     - 정수형 데이터 
#     - 예 : 투표결과, 몇명, 높낮이가 있는 수치 데이터. 
# - 범주형
#     - 가능한 범주안의 값만 사용하는 데이터
#     - A, B, C, D, F와 같은 학점 데이터
#     - 숫자가 그룹을 뜻하는 경우. 높낮이가 없는. 
# - 이진
#     - 0, 1과 같이 두개의 값을 가지는 데이터
#     - True False의 불리언 데이터
# - 순서형
#     - 값들 사이에 순위가 있는 데이터
#     - 등수를 나타내는 데이터

# #### 2. 테이블 데이터 (Pandas, 데이터프레임)
# - 레코드와 피쳐로 구성되어 있는 2차원 행렬 데이터
# - 데이터 프레임
#     - 행, 열, 값으로 구성되어 있는 데이터의 기본 구조
#     - 테이블과 같은 형태의 데이터
# - 피쳐
#     - 데이터 프레임에서 하나의 컬럼값을 의미
# - 결과, 타겟
#     - 피쳐들을 이용하여 알아내려고 하는 결과를 의미
# - 레코드
#     - 하나의 샘플 데이터를 의미

# In[4]:


titanic_df = pd.read_csv("train.csv")
titanic_df.tail(3)


# #### 3. 위치와 변이 추청
# - 데이터의 평균, 중간값, 분산, 표준편차, 백분위수, 사분위범위등을 확인하여 데이터의 특징을 확인

# In[5]:


titanic_df.describe()  #데이터가 어떻게 구성이 되어 있는지를 알수 있음. 
titanic_df.describe()  #아래 Survived의 평균 등은 의미가 없음. 


# #### 4. 그래프
# - 데이터를 표현하기에 적당한 그래프로 그려보기

# In[8]:


import seaborn as sns
sns.set()


# In[13]:


flights = sns.load_dataset("flights")
flights.tail(3)


# In[11]:





# In[14]:


flights_data = flights.pivot("month", "year", "passengers")
flights_data 


# In[5]:


sns.heatmap(flights_data, cmap="YlGnBu", annot=True, fmt="d")


# ### 5. EDA 과정을 마치고
# - EDA과정 중에 잘못된 부분을 발견하면 문제 정의, 데이터 수집, 데이터 처리로 돌아가서 다시 분석 과정을 수행
# - 데이터를 탐색한 후
#     - 수치형 데이터로 변경
#     - 원핫인코딩으로 데이터 변경
#     - 데이터를 제거

# ### 데이터의 처리(EDA)에 관한 예시
# - 수치형으로 되어 있지만 실제로는 명목형이 변수
#     - 남자, 여자 데이터가 0과 1로 표시되어 있는 데이터
#     - 카테고리 값이 숫자로 되어 있는 데이터   
# - 명목형으로 되어 있지만 실제로는 수치형인 변수
#     - 학점과 같이 A, B, C, D, F로 되어 있는 데이터
#     - 상태를 나타내는 Excellent, Good, Average, Fair, Poor과 같은 데이터
# - 합쳐서 하나로 만들수 있는 변수
#     - 성별에 대한 데이터가 두개의 컬럼으로 사용되는 데이터
#     - 예 자동차에 기어와 엔진이 같이 움직인다면, 비슷한 특징을 가진 컬럼 두개로 합킬수 있는 경우. 
# - 쪼개서 나눌수 있는 변수
#     - 요금 정보에 초대권과 무료와 입장요금의 데이터가 모두 있는 경우
# - 없는 값인지 0인지를 판별
#     - 데이터가 없어서 NaN으로 출력되면 그 값이 0인것인를 판단

# ### 프로젝트 과제   (개인과제)
# - 아래의 2개중 하나의 주제를 잡아서 간단한 데이터 분석을 해보세요.
# - 아래의 주제 말고 다른 관심있는 주제가 있다면 관심있는 주제를 분석하셔도 됩니다.
# - 

# ### 1. tips 데이터에서 팁을 많이 받으려면 어떤 데이터가 중요한지 인사이트를 찾아내세요
# - 본인이 웨이터라고 생각하고 어떻게 하면 팁을 많이 받을수 있을지 분석해 보세요.
# - 정답은 없습니다. 자유롭게 분석해보세요. 
# - 가설을 여러개 세우셔서 어떤 데이터가 가장 팁에 대한 데이터에 영향을 많이 주는지 찾아보세요.
# - 최소한 3개 이상의 가설을 세워서 분석해보세요.
# - 가설, 분석과정, 결과 확인에 대한 설명 '주석'을 달아주세요.

# In[17]:


import seaborn as sns
tips = sns.load_dataset("tips")
tips.tail()


# In[ ]:





# ### 2. 어떤 특징이 있는 다이아몬드가 비싼 다이아몬드인지 분석해보세요.
# - 본인이 보석 감별사라고 생각하고 어떤 지표가 높은 다이아몬드의 가격을 결정하는지 분석해 보세요.
# - 위의 tips와 같이 3가지 이상의 가설을 세워 분석해보세요.
# - 가설, 분석과정, 결과 확인에 대한 설명 주석을 달아주세요.

# In[19]:


diamonds = sns.load_dataset("diamonds")
diamonds.tail(2)

# 아래 테이블의 컬럼을 보고 어떤걸 뺄수 있는지


# In[ ]:


꼭 깃 써서 진행. 


# In[ ]:





# In[ ]:


# 깃 사용
1. Brach 사용

git flow

1.1. 운영
1.2. 개발
1.3. 이슈

2. Fork 뜨는것


# In[ ]:





# In[112]:


## 실습 
# 결측지 확인

titanic_df.info()


# In[113]:


# or
len(titanic_df[titanic_df["Age"].isnull()])


# In[25]:


#pip 로 설치 : pip install missingno

import missingno as msno

msno.matrix(titanic_df)  #어떤 데이터가 빠졌는지 확인 할수 있음. 맨 오른쪽이 Row 별로 보는법. 꽉체워지게끔해줘야함(전처리).


# In[114]:


msno.bar(titanic_df)  #위를 컬럼별로 보고 싶을때


# In[115]:


titanic_df.columns


# In[116]:


# 필요한 컬럼들만 남기자
# 불필요한 feature 제거. 의미없는 데이터는 드랍을 해주는것
columns = ["Survived","Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
titanic_df_1 = titanic_df[columns]


# In[117]:


msno.matrix(titanic_df_1)


# In[118]:


# AGE는 아직까지도 빈게 많다. 평균값, 중간값, 최빈값등으로 넣어서 체우면 좋다. 
# AGE는 중간값을 추가. 

# 데이터를 넣기 
sns.distplot()
titanic_df_1["Age"]

titanic_df_1.Age.dropna()  #NaN 을 없애기. 


# In[119]:


# titanic_df_1["Age"] 그래프로 그리기


sns.distplot(titanic_df_1.Age.dropna())


# In[120]:


# 중앙값 구하기
np.median(titanic_df_1.Age.dropna())
median_age = np.median(titanic_df_1.Age.dropna())
median_age


# In[121]:


# 빈곳에 중앙값 넣기 
titanic_df_1["Age"].isnull()


# In[122]:


#필터링을 해준다.
titanic_df_1[titanic_df_1["Age"].isnull()]


# In[123]:


titanic_df_1[titanic_df_1["Age"].isnull()] = median_age


# In[124]:


titanic_df_1["Age"][titanic_df_1["Age"].isnull()] = median_age 


# In[ ]:





# In[64]:


# Embarked는 row 삭제를 통해 전처리를 한다. 

titanic_df_1.dropna(inplace=True)


# In[67]:


titanic_df_1.reset_index(drop=True, inplace=True)


# In[68]:


titanic_df_1.tail(2)


# In[96]:


# 성별 one-hot encoding
import pandas as pd


# In[97]:


data1 = pd.get_dummies(titanic_df_1["Sex"])
data2 = pd.get_dummies(titanic_df_1["Embarked"])


# In[98]:


titanic_df_2 = pd.concat([titanic_df_1, data1, data2], axis=1)
titanic_df_2.tail(2)


# In[84]:


titanic_df_2.drop(columns=["Sex","Embarked"], inplace=True)


# In[99]:


titanic_df_2.tail(2)


# In[100]:


# AGE : 연령대 
titanic_df_2["Ages"] = (titanic_df_2["Age"]//10 *10).astype("int")

titanic_df_2.tail(2)


# In[ ]:





# In[ ]:


# Adult : 성인 여부 컬럼 추가 (성인이면 1, 미성년이면 0)


# In[103]:


titanic_df_2["Adult"] = 0
titanic_df_2.tail()


# In[102]:


#titanic_df_2.loc[ 로우 , 컬럼]

titanic_df_2.loc[titanic_df_2["Ages"] >= 20, "Adult"]   #위 885 가 없음


# In[105]:


titanic_df_2.loc[titanic_df_2["Ages"] >= 20, "Adult"] = 1 

titanic_df_2


# In[ ]:





# In[ ]:


# 데이터 샘플 

http://kosis.kr/index/index.do
    
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
    

