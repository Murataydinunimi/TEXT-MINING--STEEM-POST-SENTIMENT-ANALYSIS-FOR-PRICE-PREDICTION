#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#Base and Cleaning 
import json
import requests
import pandas as pd
import numpy as np
import emoji
import regex
import string
from collections import Counter


import re
import numpy as np
import pandas as pd
from pprint import pprint
from random import seed
import random
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



import spacy.cli
#spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
# NLTK Stop words
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

from scipy.stats import pearsonr
import statsmodels.api as sm


# In[ ]:


# data preprocessing / data cleaning
import re
import string
import emoji

def clean_text(text):
   
 
#Make text lowercase   
    text = str(text).lower()
#remove text in square brackets
    text = re.sub(r'\[.*?\]', '', text)
#remove punctuation   
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
#remove words containing numbers
    text = re.sub(r'\w*\d\w*', '', text)
#remove links   
    text = re.sub(r'http\S+', '', text)
# html 
    text = re.sub(r'html','',text)
    
#email
    text = re.sub(r'\S*@\S*\s?','',text)
    
# Remove new line characters
    
    text = re.sub(r'\s+',' ',text)
    
# Remove distracting single quotes
    
    text = re.sub("\'", "", text)

                   
                   
    
#remove emojis    
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'', text)
    
    return text


def give_emoji_free_text(text):
    """
    Removes emoji's from tweets
    Accepts:
        Text (tweets)
    Returns:
        Text (emoji free tweets)
    """
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def url_free_text(text):
    '''
    Cleans text from urls
    '''
    text = re.sub(r'http\S+', '', text)
    return text

# Apply the function above and get tweets free of emoji's
call_emoji_free = lambda x: give_emoji_free_text(x)


def href_free_text(text):
    '''
    Cleans text from html
    '''
    text = re.sub(r'<a href\S+', '', text)
    return text

def img_free_text(text):
    
    text = re.sub(r'img', '', text)
    return text
    
def div_free_text(text):
    
    text = re.sub(r'div', '', text)
    return text


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)



# In[4]:


def clean_data(data,column,new_column="cleaned_text"):
 
    
    print("Shape of given data:",data.shape)

    
    data[new_column] = data[column].apply(clean_text)
    
    print("Text is being cleaned from punctuations, brackets, numbers,links, emails")
    
    data[new_column] = data[new_column].apply(call_emoji_free)
    
    print("Text is being cleaned from emojis")
    
    data[new_column] = data[new_column].apply(url_free_text)
    
    print("Text is being cleaned from urls")
    
    data[new_column] = data[new_column].apply(href_free_text)
    
    print("Text is being cleaned from html")
    
    data[new_column] = data[new_column].apply(img_free_text)
    data[new_column] = data[new_column].apply(div_free_text)
    data[new_column] = data[new_column].apply(remove_tags)
    
    print("Text is being cleaned from tags")
    
    print("Shape of output data:",data.shape)
    
    return data


# In[5]:


def drop_empty_strings(df,column):

    
    print("Shape of given data:",df.shape)
    
    drop_indices = []

    for i in range(len(df)):

        len_str = len(df[column].loc[i].strip())

        if len_str==0:

            drop_indices.append(i)
    
    print("Number of rows to be dropped : {}".format(len(drop_indices)))
    
    if len(drop_indices) == 0:
        
        print("There is no empty text")
            
    df = df.drop(drop_indices, axis=0).reset_index()
    df = df.drop("index",axis=1)
    #check if after dropping the rows, the below equality holds
    
    assert df.shape[0] -len(drop_indices) == len(df) - len(drop_indices)

    print("Shape of output data:",df.shape)
    
    return df


# In[6]:


def get_language(df,column):
    
  
    
    print("Shape of given data:",df.shape)
    
    
    from langdetect import detect
    lang_dict = {}
    drop_rows = []

    for i in range(len(df)):
        try:
            language = detect(df[column].loc[i])
            
            if language == "zh-cn":
                
                language= "zh_cn"
                
            lang_dict[i] = language
        except:
            #print("Unable to detect the language. The row {} will be dropped".format(i))
            drop_rows.append(i)
    
    print("Unable to detect language for {} rows. They will be dropped".format(len(drop_rows)))
            
    lang = pd.DataFrame.from_dict(lang_dict,orient="index").rename({0:"Lang"},axis=1)
    df = pd.concat([df,lang],axis=1)
    df = df.drop(drop_rows,axis=0).reset_index().drop("index",axis=1)

    
    print("Number of languages found: {}".format(len(np.unique(df.Lang))))
    
    print("Shape of output data",df.shape)
    

    return df


# In[7]:


def get_eng_post(df,column="Lang"):
    
 
    
    print("Shape of given data",df.shape)
    
    df = df[df[column] =="en"]
    df = df.reset_index().drop("index",axis=1)
    
    print("Shape of output data", df.shape)
    
    return df


# In[8]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[9]:


def pre_processing(df,column,new_column):
    
    print("----Data Cleaning----")
    df = clean_data(df,column,new_column="cleaned_text")
    print("----Dropping empty text----")
    df = drop_empty_strings(df,new_column)
    print("----Language is being assigned----")
    df = get_language(df,new_column)
    print("----Extracting Posts in English---")
    df = get_eng_post(df,column="Lang")
    
    return df


# In[10]:


def prepare_to_LDA(df,column,tokenizer=sent_to_words, remove_stopwords = remove_stopwords, make_bigrams = make_bigrams,
                  lemmatization = lemmatization):
    
    sentences = df[column].values.tolist()
    
    
    #tokenize
    
    data_words = list(tokenizer(sentences))
    
    print("---- Posts are tokenized----")
    
    #build the bigram
    
    
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
    bigram_mod = gensim.models.phrases.Phraser(bigram)

        
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    
    print("----Stopwords are being removed----")

    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)
    
    print("----Bigrams are created----")

   
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    print("----Tokens are lemmatized---")
    
    return data_lemmatized

    
    


# In[11]:


def modelling_LDAmallet(data_lemmatized, n_topics, random_state, chunksize, passes,model=["LDA","MALLET"]):
    
    
    print("----Corpus of tokens are created----")


    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    
    print("----{} object is being created".format(model))
    
    if model == "LDA":
        
        coherence_values = []
        model_list = []
        
        print("----Model LDA will return six objects : topics, perplexity, coherence, model object,corpus,id2word----")
        print("----Metrics are being calculated----")
        
        for n_topic in n_topics:


            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=n_topic, 
                                                   random_state=random_state,
                                                   update_every=1,
                                                   chunksize=chunksize,
                                                   passes=passes,
                                                   alpha='auto',
                                                   per_word_topics=True)


            

            topics = lda_model.print_topics()
            perplexity = lda_model.log_perplexity(corpus)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            coherence_values.append(coherence_lda)
            model_list.append(lda_model)


        
        
        return topics, perplexity, coherence_values,model_list,corpus,id2word
    
    
    else:
        coherence_values = []
        model_list = []
        
        print("----Model MALLET will return five objects: topics, coherence, model object, corpus, id2word----")

        print("----Metrics are being calculated----")
        
        for n_topic in n_topics:
        

            mallet_path = 'mallet-2.0.8/bin/mallet'
            ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=n_topic, id2word=id2word)
            topics_mallet = ldamallet.show_topics(formatted=False)
            coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_ldamallet = coherence_model_ldamallet.get_coherence()
            coherence_values.append(coherence_ldamallet)
            mallet_lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
            model_list.append(mallet_lda_model)
       

        
        return topics_mallet, coherence_values,model_list,corpus,id2word

    
    


# In[12]:


def compute_coherence_values(dictionary, corpus, limit, texts, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[13]:


def format_topics_sentences(ldamodel, corpus, texts):
    
    print("Topics are being assigned based on the MALLET results.")

    # Init output
    sent_topics_df = pd.DataFrame()
    #k=0

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
        
        #k+=1
        #print(k)
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    return(df_dominant_topic)


# In[14]:


def calculate_dailytopic_growth(data,df_dominant_topic):
    
    res = data[["timestamp","cleaned_text"]].copy()
    res["Topic"] = df_dominant_topic["Dominant_Topic"].copy()
    res["timestamp"] = pd.to_datetime(res["timestamp"])
    res["year"] = res["timestamp"].dt.year
    res["month"] = res["timestamp"].dt.month
    res["day"] = res["timestamp"].dt.day
    
    day_and_topic =  res.groupby(["year","month",'day',"Topic"]).agg({"Topic":["count"]}).reset_index()
    day_and_topic.columns = [''.join(col) for col in day_and_topic.columns.values]
    
    day_and_topic["Relatives"] = 1
    
    counts = {}

    for each_year in np.unique(day_and_topic["year"]):
    
        counts[each_year]= {}

    
        for each_month in np.unique(day_and_topic[day_and_topic["year"]==each_year]["month"]):
        
            counts[each_year][each_month] = {}
        
            for each_day in np.unique(day_and_topic[(day_and_topic["year"]==each_year) &
                                                      (day_and_topic["month"]==each_month)]["day"]):
            
                countie = day_and_topic[(day_and_topic["year"]==each_year) & 
                          (day_and_topic["month"]==each_month) & (day_and_topic["day"]==each_day)].Topiccount.sum()
            
                counts[each_year][each_month][each_day] = countie
                
                indices = day_and_topic[(day_and_topic["year"]==each_year) & 
                          (day_and_topic["month"]==each_month) & (day_and_topic["day"]==each_day)].index.to_list()
                
                for ind in indices:
        
                    day_and_topic.at[ind,"Relatives"] = day_and_topic["Topiccount"][ind]/counts[each_year][each_month][each_day]
            
    daily_topics = []
    for topic in range(len(np.unique(day_and_topic["Topic"]))):

        topic_df = day_and_topic[day_and_topic["Topic"]==topic].reset_index().drop("index",axis=1)

        topic_df["Daily_growth"] = topic_df["Relatives"].pct_change()

        topic_df.fillna(topic_df["Daily_growth"].mean(), inplace=True)

        daily_topics.append(topic_df)
        
    return daily_topics
        
        


# In[15]:

def calculate_correlation(daily_topics,steem_daily_growth,min_date,max_date):
    
    corr_per_topic =[]
    
    for i in range(len(daily_topics)):
        
        data_time_stamp = daily_topics[i][["year","month","day","Daily_growth"]].copy()
        data_time_stamp.columns = ["year","month","day","topic_growth"]
        data_time_stamp["TimeStamp"] = pd.to_datetime(data_time_stamp[['year', 'month', 'day']])
        data_time_stamp = data_time_stamp.drop(["year","month","day"],axis=1)                
        sample = pd.concat([steem_daily_growth["Daily_growth"],data_time_stamp],axis=1)
        sample = sample.reset_index()
        sample = sample.drop("index",axis=1)
        sample = sample[(sample["TimeStamp"]>=min_date) & (sample["TimeStamp"]<=max_date)]
        
        corr_per_topic.append(sample)
        
        steem_daily_growth["timestamp"] =  pd.to_datetime(steem_daily_growth[['year', 'month', 'day']])
        
        steem_daily_growth = steem_daily_growth[(steem_daily_growth["timestamp"]>=min_date) &(steem_daily_growth["timestamp"]<=max_date)]
        
                                                          
        
        
    corr_res = {}

    for i in range(len(daily_topics)):
        try:
            pea_corr,p_val = pearsonr(steem_daily_growth['Daily_growth'], corr_per_topic[i]["topic_growth"])
            corr_res[pea_corr] = p_val
        except:
                    print("Pearson correlation can not be calculated for topic {}. In some of the days this topic is never used, so the growth rate will be NaN.".format(i))
        
    corr_df = pd.DataFrame.from_dict(corr_res,orient="index").reset_index().rename({"index":"Pearson_corr",0:"P_val"},axis=1)
    corr_df.index.name = "Topics"

        
    return corr_per_topic,corr_df


# In[16]:


def steem_growth(price_data):
    
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    cols = ["Open*","High","Low","Close**"]
    
    #convert to float

    for column in cols:
        float_type = pd.DataFrame([float(str(i).replace(",", "")) for i in price_data[column]]).astype("float")/100
        price_data = price_data.drop(column,axis=1)
        price_data = pd.concat([price_data,float_type],axis=1).rename({0:column},axis=1)
        
        
    #calculate growth rate
    
    price_data = price_data.sort_values(by="Date").reset_index().drop("index",axis=1)
    price_data["Daily_growth"] = price_data["Close**"].pct_change()
    price_data.fillna(price_data["Daily_growth"].mean(),inplace=True)
    price_data["year"] = price_data.Date.dt.year
    price_data["month"] = price_data.Date.dt.month
    price_data["day"] = price_data.Date.dt.day
    price_data = price_data.drop(["Date","Volume","Market Cap","Open*","High","Low","Close**"],axis=1)
    

    return price_data


# In[17]:
def plot_coh_val(n_topics,coherence_values):

    plt.plot(n_topics, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


# In[18]:

def compute_cross_cor(corr_per_topic):
    
    n_topics = len(corr_per_topic)
    
    for i in range(n_topics):



        res = sm.tsa.stattools.ccf(corr_per_topic[i]["Daily_growth"],corr_per_topic[i]["topic_growth"])
        n_lags = len(res)
        conf_level = 2 / np.sqrt(n_lags)


        plt.figure(figsize=(12,7), dpi= 80)

        plt.hlines(0, xmin=0, xmax=100, color='gray')  # 0 axis
        plt.hlines(conf_level, xmin=0, xmax=100, color='gray')
        plt.hlines(-conf_level, xmin=0, xmax=100, color='gray')

        plt.bar(x=np.arange(len(res)), height=res, width=.3)


        plt.title('Cross Correlation Plot: Steem-Price vs Topic_{}'.format(i), fontsize=22)
        plt.xlim(0,len(res))
        plt.show()


def plot_series(corr_per_topic):
    
    n_topics = len(corr_per_topic)
    
    for i in range(n_topics):
        
        corr_per_topic[i].plot(x='TimeStamp', y=['Daily_growth','topic_growth'],figsize=(10, 5), grid=True)
        plt.title('Steem-Price vs Topic {} Usage Daily Growth'.format(i), fontsize=22)

def get_topics(df_dominant_topic):
    
    topics_keywords = {}

    for topic in range(len(np.unique(df_dominant_topic.Dominant_Topic))):
        topics = list(np.unique(df_dominant_topic[df_dominant_topic.Dominant_Topic==topic].Keywords))
        topics_keywords[topic] = topics

    return topics_keywords


def random_sample(df, new_len, random_state = 1):
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.day
    
    days = range(1,len(np.unique((df["day"])))+1)
    info_dict = {}
    new_indices = []
    random.seed(random_state)
    
    for day in days:
    
        len_dat = len(df[df["day"]==day])
        
        info_dict[day] = [len_dat,len_dat/len(df)]
        
        relative_usage = info_dict[day][1]
        
        ind_list = df[df["day"]==day].index.tolist()
        
        new_indices.append(random.sample(ind_list,int(new_len*relative_usage)))
        
    flat_list = [item for sublist in new_indices for item in sublist]
    
    sample = df.loc[flat_list].sort_values(by="timestamp").reset_index().drop("index",axis=1).copy()
    
    return sample
        
        
        
        
def read_data (year,month):
    import os
    year = str(year)
    month = month.lower()
    path= os.getcwd() + '/'+month+"_"+year
    df_dominant_topic = pd.read_csv(path+"/df_dominant_topic_"+month+".csv")
    clean_df = pd.read_csv(path+"/sampled_clean_eng_"+month+".csv")
    
    assert ((clean_df.shape)==(df_dominant_topic.shape))

    clean_df["Topic"] = df_dominant_topic["Dominant_Topic"].copy()
    
    
    MIN_DATE = str(min(clean_df["timestamp"])).split(' ')[0]
    MAX_DATE = str(max(clean_df["timestamp"])).split(' ')[0]
    
    price_data = pd.read_csv("SteemCoinFull.csv",sep=";")
    price_data["Date"] = pd.to_datetime(price_data["Date"])

    price_data = price_data[(price_data["Date"]>=MIN_DATE) & (price_data["Date"]<=MAX_DATE)]
    price_data = price_data.sort_values(by="Date")
    price_data = price_data.reset_index()
    price_data = price_data.drop("index",axis=1)

    
    return clean_df, df_dominant_topic,price_data


def map_topics(df_dominant_topic,corr_df):
    
    topic_no = len(get_topics(df_dominant_topic))
    topics_list = []
    
    for i in range(topic_no):
        topics = str(list(get_topics(df_dominant_topic)[i]))
        
        topics = (topics[len("'")+1:-len("'")-1].split(',')[:4])
        
        topics = [topic.strip() for topic in topics]
        
        topics = str(i)+"_"+"_".join(topics)
        
        topics_list.append(topics)
        
    corr_df = corr_df.set_axis(topics_list, axis=0)
        
    return corr_df

        
def find_significance(mapped_topics):
    
    p_val_list = mapped_topics.P_val.to_list()
    
    for i in range(len(mapped_topics)):
        
        if p_val_list[i] < .10:
            
            print("Topic {} is significant at %{} level".format(mapped_topics.index[i], p_val_list[i]))
            
            
            
def calculate_polarity(data_to_polarity):
    
    import tqdm
    
    res_dict = {}
    text = data_to_polarity.cleaned_text.tolist()
    
    for index in tqdm.tqdm(range(len(text))):
        
        res = predict(text[index])["score"]
        res_dict[index] = res
    
    result = pd.DataFrame.from_dict(res_dict,orient="index")
    result = result.rename({0:"score"},axis=1)
    data_to_polarity["score"] = result["score"].copy()
    
    data_to_polarity["timestamp"] =pd.to_datetime(data_to_polarity["timestamp"].apply(lambda row: row.split(' ')[0]))
        
    
    
    return data_to_polarity
    
    
            
        


def prepare_data_LSTM(month,year):
    
    import os
    
    month = month.lower()
    year = str(year)
    
    topics_path = os.getcwd()+"/polarity_results"+"/mapped_topics_"+month+"_"+year+".csv"
    
    
    polarity_path = os.getcwd()+"/polarity_results"+"/polarity_"+month+"_"+year+".csv"
    
    steem_path =  os.getcwd()+"/polarity_results"+"/SteemCoinFull.csv"
    
    polarity_res = pd.read_csv(polarity_path)
    
    topics = pd.read_csv(topics_path)
    
    topics.columns = ["Topics","Pearson_corr","P_val"]
    
    topics_codes = list(np.unique(polarity_res["Topic"]))
    
    list_of_topics = (list(topics.Topics.values))
    
    indices_to_get = []

    for i in range(len(list_of_topics)):
    
        if i > 9:
        
            inty = int(list_of_topics[i][:2])
        
        else:
        
            inty = int(list_of_topics[i][0])
            
            
        
        if inty in topics_codes:

            indices_to_get.append(i)
        
        
    topic_columns = list(topics.loc[indices_to_get].Topics.values)
    
    datas_to_concat = []
    
    for ind,value in enumerate(topics_codes):
        
        res = polarity_res[polarity_res["Topic"]==value].groupby(by=["timestamp"]).agg({"score":"mean"}).reset_index().copy()
        
        res = res.rename({"score":topic_columns[ind]},axis=1)
        
        datas_to_concat.append(res)
        
            
    df_1 = pd.merge(datas_to_concat[0],datas_to_concat[1], on = "timestamp")
    df_2 = pd.merge(datas_to_concat[2],datas_to_concat[3], on = "timestamp")

    df_to_join_steem = pd.merge(df_1,df_2, on = "timestamp")
    df_to_join_steem["timestamp"] = pd.to_datetime(df_to_join_steem["timestamp"])

        
    MIN_DATE = min(res["timestamp"])
    MAX_DATE = max(res["timestamp"])
    
    price_data = pd.read_csv("SteemCoinFull.csv",sep=";")
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data.columns = ["timestamp","open","high","low","close","volume","market_cap"]

    price_data = price_data[(price_data["timestamp"]>=MIN_DATE) & (price_data["timestamp"]<=MAX_DATE)]
    price_data = price_data.sort_values(by="timestamp")
    price_data = price_data.reset_index()
    price_data = price_data.drop("index",axis=1)
    price_data = price_data.drop(["open","high","low","volume","market_cap"],axis=1)

    
    final_df = pd.merge(df_to_join_steem,price_data,on="timestamp")
    
    final_df["close"] = final_df["close"].str.replace(',', '.').astype(float)
    
    final_df=final_df.set_index("timestamp")
    
    
    return final_df
    


        
        
        
        
        
    
    
    

    
        
    
    
    
    
    
    

