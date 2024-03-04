# installing packages
!pip install numpy
!pip install pandas
!pip install surprise
!pip install nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv("/content/sample_data/hotels.csv", encoding="latin1")
# check out the shape and top few observations of the data
data.shape
data.head(2)
# count the common length of description
data['desc_length'] = data.desc.apply(lambda x:len(x.split(" ")))
data.desc_length.describe()

print("There are {} documents in total".format(data.desc_length.describ
e()['count']))
print("The longest document has {} words".format(data.desc_length.descr
ibe()['max']))
print("The shortest document has {} words".format(data.desc_length.desc
ribe()['min']))
# let's check out whether there is any duplicates
len(data.name.unique())
# getting the word frequency of the description
word_freq = data.desc.str.split(expand=True).stack().value_counts()
word_freq[:20]
# create a bar graph of the word frequency
word_freq_top_20 = dict(word_freq[:20])
plt.figure(figsize = (14,14))
plt.bar(range(len(word_freq_top_20)), word_freq_top_20.values(), tick_l
abel=list(word_freq_top_20.keys()))
import nltk
nltk.download('stopwords')
# stop word removal
stop_words = set(stopwords.words('english'))
stop_words
# we will then remove stop words from the hotel description to clean up
the data
data = data[['name','address','desc','desc_length']]
data.head()
# we will do some data cleaning
data['desc_lower'] = data.desc.apply(lambda x:x.lower())
data['without_stopwords'] = data['desc_lower'].apply(lambda x: ' '.join
([word for word in x.split() if word not in (stop_words)]))
data.head()
# comparing description with stopwords and description without stopwords
print(data.without_stopwords[0])
print('------------------------')
print(data.desc[0])
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def clean_text(text):
 """
 text: a string
 
 return: modified initial string
 """
 text = REPLACE_BY_SPACE_RE.sub(' ', text)
# replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the
matched string in REPLACE_BY_SPACE_RE with space.
 text = BAD_SYMBOLS_RE.sub('', text)
# remove symbols which are in BAD_SYMBOLS_RE from text. substitute the
matched string in BAD_SYMBOLS_RE with nothing.
 return text
 
data['desc_complete_cleaned'] = data['without_stopwords'].apply(clean_t
ext)
# compare the cleaned version of description
data['desc_complete_cleaned'][0]
word_freq_clean = data.desc_complete_cleaned.str.split(expand=True).sta
ck().value_counts()
word_freq_clean[:20]
plt.figure(figsize=(16,16))
plt.bar(range(len(word_freq_clean[:20])), dict(word_freq_clean[:20]).va
lues(), tick_label=list(dict(word_freq_clean[:20]).keys()))
data.set_index('name', inplace = True)
# calculate cosine similarity between documents - we use tfidf
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, sto
p_words='english')
tfidf_matrix = tf.fit_transform(data['desc_complete_cleaned'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(similarity_matrix)
indices = pd.Series(data.index)
indices[indices == "Vermont Inn"].index[0]
def recommendations(name, similarity_matrix = similarity_matrix):
 
 recommended_hotels = []
 
 # gettin the index of the hotel that matches the name
 idx = indices[indices == name].index[0]
 # creating a Series with the similarity scores in descending order
 score_series = pd.Series(similarity_matrix[idx]).sort_values(ascend
ing = False)
 # getting the indexes of the 10 most similar hotels except itself
 top_10_indexes = list(score_series.iloc[1:11].index)
 
 # populating the list with the names of the top 10 matching hotels
 for i in top_10_indexes:
 recommended_hotels.append(list(data.index)[i])
 
 return recommended_hotels
# get the list of optional names
data.index
recomendations = input("what types of hotels would you like me to recom
mend for ya?")
recommendations(recomendations)
