#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[8]:


df = pd.read_csv("D:\data analysis\Reviews.csv")


# In[9]:


df.head()


# In[10]:


print(df.shape)


# In[11]:


df = df.head(500)
print(df.shape)


# In[12]:


#quick eda


# In[13]:


df['Score'].value_counts().sort_index().plot(kind='bar', title='count of reviews by stars', figsize=(10,5))

plt.xlabel('Review Stars')
plt.show()


# In[14]:


#basic nltk


# In[15]:


example = df['Text'][50]
print(example)


# In[16]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[17]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[18]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[19]:


#STEP 1. VADER seniment scoring


# In[20]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[21]:


sia.polarity_scores('I am so happy!')


# In[22]:


sia.polarity_scores('This is the worst thing ever')


# In[23]:


sia.polarity_scores(example)


# In[24]:


res = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[25]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index':'Id'})
vaders = vaders.merge(df, how='left')


# In[26]:


#now we have sentiment score and metadata
vaders


# In[27]:


#plot vaders results


# In[28]:


sns.barplot(data= vaders, x='Score', y='compound')
plt.title('compound score by amazon star review')
plt.show()


# In[29]:


fig, axs = plt.subplots(1,3, figsize=(12,3))
sns.barplot(data= vaders, x='Score', y='pos', ax= axs[0])
sns.barplot(data= vaders, x='Score', y='neg', ax= axs[1])
sns.barplot(data= vaders, x='Score', y='neu', ax= axs[2])
axs[0].set_title('positive')
axs[1].set_title('negative')
axs[2].set_title('neutral')
plt.show()


# In[30]:


#Step 3. Roberta pretrained model


# In[31]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[32]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[33]:


print(example)
sia.polarity_scores(example)


# In[34]:


#run on roberta


# In[45]:


encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores =  softmax(scores)
score_dict = {
    "roberta_neg" : scores[0],
    "roberta_neu" : scores[1],
    "roberta_pos" : scores[2]
}
print(score_dict)


# In[51]:


def polarity_scores_roberta(example):
    encoded_text= tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores =  softmax(scores)
    score_dict = {
    "roberta_neg" : scores[0],
    "roberta_neu" : scores[1],
    "roberta_pos" : scores[2]
     }
    return score_dict


# In[71]:


res = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_results_rename = {}
        for key,value in vader_result.items():
           vader_results_rename[f"vader_{key}"] = value
        roberto_result = polarity_scores_roberta(text)
        both = {**vader_results_rename, **roberto_result}
        res[myid] = both
    except RuntimeError:
            print(f'broke for id{myid}')


# In[68]:


both


# In[74]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index':'Id'})
results_df = results_df.merge(df, how='left')


# In[75]:


results_df.head()


# In[76]:


#compare


# In[79]:


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos', 
                  'vader_compound' , 'roberta_neg', 
                  'roberta_neu', 'roberta_pos'], 
            hue = 'Score',
            palette='tab10')
plt.show()


# In[80]:


#review examples


# In[83]:


results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[84]:


results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[85]:


results_df.query('Score == 1').sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[86]:


results_df.query('Score == 1').sort_values('vader_neg', ascending=False)['Text'].values[0]


# In[ ]:




