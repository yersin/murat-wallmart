# using nltk lib to do lda
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords

print "extended stopped words list we use: "
print stopwords.words("english")

print "load csv file ... "
data = pd.read_csv("../k-means/walmart-import-data-full.csv")
for name in enumerate(data.columns):
    print name
#print " ".join(data.PD)

#tokens = word_tokenize(data.PD[0])
#print tokens
#print re.findall(r'[a-zA-Z]+', data.PD[0].lower())

print "data cleaning ..."
stopword = pd.read_csv("stopwords2.csv")
st = stopword.Stop
stw = st[0]
for i in st:
    stw = stw + i
tokens = [" ".join([element for element in re.findall(r"[a-zA-Z']{2,}", data.PD[0].lower()) if element not in stw])]
print tokens
for i in range(1,10000):
    if i % 1000 == 0:
        print i
    if data.PD[i] == data.PD[i]:
        tokens = tokens + [" ".join([element for element in re.findall(r"[a-zA-Z']{2,}", data.PD[i].lower()) if element not in stw])]
print tokens
from collections import Counter
count = Counter(tokens)
print count.most_common(20)

#tk_stop = [element for element in tokens if element not in stw]
#print tk_stop
#count = Counter(tk_stop)
#print count.most_common(100)

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#print "1"
vec = CountVectorizer(decode_error ='ignore', token_pattern = "[a-zA-Z]{3,}", lowercase=True, stop_words = stopwords.words('english'), max_features = 100)
#tks = " ".join(tk_stop)
#print "2"
#print vec
#print tks
#sparse_matrice = vec.fit_transform(tk_stop)
sparse_matrice = vec.fit_transform(tokens)
print "training start"
X_train, X_test = train_test_split(sparse_matrice, test_size=0.20, random_state=42) ## 80 -20 spl
lda = LatentDirichletAllocation(n_topics = 20, max_iter=1500, topic_word_prior= 0.01, doc_topic_prior=1.0,learning_method='batch',n_jobs= 1,random_state=275)
topic_components = lda.fit_transform(X_train)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

print_top_words(lda,vec.get_feature_names(), 10)
print "lda perplexity: ", lda.perplexity(X_test)
