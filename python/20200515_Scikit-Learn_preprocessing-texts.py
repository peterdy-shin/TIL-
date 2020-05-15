#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import missingno as msno
font_list = fm.findSystemFonts(fontpaths = None, fontext = 'ttf')
font_list[:]

plt.rc('font', family='AppleGothic') # For Windows
print(plt.rcParams['font.family'])


# In[ ]:


# Bag of Words
 문서(d1, d2,...)를 숫자벡터로 변환하는것.
단어장(t1, t2..)를 만듦. 
di 에 해당하는 단어들이 포함되어있는지 확인하는 방법. 
xij = 문서 di 내에 단어 tj의 출혈빈도 
xij = 1   # 단어가 있으면
xij = 0   # 단어가 없으면


# In[ ]:


scikit learnd에서는 문서 전처리용 클래스 4개를 제공함. 


# # 1. DictVectorizer

# In[3]:


# 1. DictVectorizer
# 최대한 간단한 형태의 vectorizer (문서를 벡터화 하니까 vectorizer)

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)

# 몇 번 나왔는지 수동으로 넣어야함. 
# D 안에는 리스트가 있고 리스트안에는 딕셔너리 두개가 존재. 각 딕셔너리 하나가 문서 하나를 뜻함. 
# 첫번째 문서에는 A 라는 단어 1번, B라는 단어 2번 쓰임. 두번째 문서에는 B라는 단어 3번, C라는 단어 1번 
D = [{'A':1, 'B':2}, {'B':3, 'C':1}] 

# 두개의 벡터가 나옴. 
X = v.fit_transform(D)
X


# In[4]:


# vocabulary : 첫번째 단어는 A, B는 두번째 단어
# 위에 숫자들의 순서에 따라 알파벳 빈도수를 알려줌. 

v.feature_names_


# In[5]:


# 아예 없었던 D를 넣게 되면, 표기를 안해줌.
# 따라서 처음에 corpus를 넣을때 모든 단어들을 넣어야한다. 
v.transform({'C':4, 'D':3})


# # 2. CountVectorizer
# 

# In[ ]:


# 카운트까지 해줌. 
# 문서를 넣어주기만 하면 됨. 
1. 문서를 토큰 리스트로 변환하고
2. 문서내 각 단어의 빈도수를 세어 주고
3. 각 문서를 BOW 인코딩 벡터로 변환 


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',  #1번 문서
    'This is the second document.',
    'This is the third one.'
    'is this the first document?',
    'The last document?',          #5번 문서
]
vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_   # vocab 단어장을 만들어줌. 


# In[11]:


# 위에서 숫자는 해당 단어의 순서를 말해줌. 
# this 는 8번째 단어. 몇번째 단어인지.


# In[12]:


vect.transform(['This is the second document.']).toarray()

# 0 번째 : document 라는 단어 출연. 1
# 1 번째 : first 라는 단어는 출연하지 않음. 0
# 등등


# In[13]:


# vocab에 없는 단어가 들어가면? 아예 없이 나온다. 
# 따라서 맨처음 corpus가 중요함. 
vect.transform(['Something completely new']).toarray()


# # 2.1. Stop Words 인수 사용

# In[15]:


# and, is , the, 등 불필요한 단어를 없앰. 
vect = CountVectorizer(stop_words=["and", "is", "the", "this"]).fit(corpus)
vect.vocabulary_


# In[17]:


#english를 넣을 경우, 선별된 단어들을 다 넣어줌.
vect = CountVectorizer(stop_words="english").fit(corpus)
vect.vocabulary_


# # 2.2. 토큰 생성시 인수 사용

# In[ ]:


# analyzer, tokenizer, token_pattern 인수로 사용할 토큰 생성기를 선택 가능 


# In[19]:


# analyzer 인수에서 Character 단위로
vect = CountVectorizer(analyzer="char").fit(corpus)
vect.vocabulary_


# In[22]:


# token_pattern 인수에서 정규표현식을 활용해서 분류 
vect = CountVectorizer(token_pattern="t\w+").fit(corpus)
vect.vocabulary_

# t 로 시작하고 알파벳으로 되어 있는 단어 갯수 파악


# In[23]:


# 외부 토크나이저를 사용하고 싶을때
import nltk

vect = CountVectorizer(tokenizer=nltk.word_tokenize).fit(corpus)
vect.vocabulary_


# # N 그램

# In[24]:


# n 그램 = 단어장 생성에 사용할 토큰의 크기를 결정.
# 모노 그램 = 토큰 하나만 단어로 사용함.
# 바이 그램 = 두개의 연결된 토큰을 하나의 단어로 사용함. 


vect = CountVectorizer(ngram_range=(2,2)).fit(corpus)  # 바이그램 사용시. 두 개의 단어로 끊어서
vect.vocabulary_
# 단어가 아니라 구(phrase)로 토큰화하겠다라는 듯. 


# In[26]:


# t 로 시작하는 단어중에서 모노, 바이그램 사용시.
vect = CountVectorizer(ngram_range=(1,2), token_pattern="t\w+").fit(corpus)  
vect.vocabulary_


# # 빈도수 

# In[ ]:


# stopwords를 쓰는 이유 - 여러 문서에 공통적으로 여러번 쓰이는 영양가 없는 단어를 빼기 위해 
max_df, min_df 명령어로 빈도를 설정해서, 일정 횟수이상 쓰이는 단어만 볼수 있게 할수 있음. 

max_df 로 지정한 값을 초과 하거나
min_df 로 지정한 값보다 작을 경우 무시. #너무 적게 쓰인 애들도 의미가 없으니 빼겠다. 


# In[27]:


vect = CountVectorizer(max_df=4, min_df=2).fit(corpus)
vect.vocabulary_, vect.stop_words_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




