##############
# k means
##############

ClusterNUM = 400			# the number of clusters
END = 10000 				# the number of data encountered
output_filename='o.csv'		# the name of output file
data_path = "walmart-import-data-full.csv"
stopword_path = "stopwords2.csv"

####################################################
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords

print "load csv file ... "
data = pd.read_csv(data_path)
for name in enumerate(data.columns):
    print name
#print " ".join(data.PD)

#tokens = word_tokenize(data.PD[0])
#print tokens
#print re.findall(r'[a-zA-Z]+', data.PD[0].lower())

print "data cleaning ..."
stopword = pd.read_csv(stopword_path)
st = stopword.Stop
stw = st[0]
for i in st:
    stw = stw + i
tokens = [" ".join([element for element in re.findall(r"[a-zA-Z']{2,}", data.PD[0].lower()) if element not in stw])]
#print tokens
for i in range(1,END):
    if i % 1000 == 0:
        print i
    if data.PD[i] == data.PD[i]:
        tokens = tokens + [" ".join([element for element in re.findall(r"[a-zA-Z']{2,}", data.PD[i].lower()) if element not in stw])]

#print tokens

print "Common words"
from collections import Counter
count = Counter(tokens)
print count.most_common(20)

###########################################################
# vecotrize
#########################################################
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
#print "1"
vec = CountVectorizer(decode_error ='ignore', token_pattern = "[a-zA-Z]{3,}", lowercase=True, stop_words = stopwords.words('english'), max_features = ClusterNUM)
#tks = " ".join(tk_stop)
#print "2"
#print vec
#print tks
#sparse_matrice = vec.fit_transform(tk_stop)
sparse_matrice = vec.fit_transform(tokens)
print "training start"
# X_train, X_test = train_test_split(sparse_matrice, test_size=0.20, random_state=42) ## 80 -20 spl
X_train = sparse_matrice
labels = vec.get_feature_names()
####################################
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=ClusterNUM, random_state=0).fit(X_train)
print "labels:"
labels = kmeans.labels_
print labels
print "labels shape:"
print kmeans.labels_.shape
print "centorids:"
print kmeans.cluster_centers_

# from sklearn import metrics
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmeans.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, kmeans.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmeans.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, kmeans.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, kmeans.labels_, sample_size=1000))

print()


# if not opts.use_hashing:
print("Top terms per cluster:")

# if opts.n_components:
	# original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
	# order_centroids = original_space_centroids.argsort()[:, ::-1]
# else:
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vec.get_feature_names()
for i in range(ClusterNUM):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

print "score"
print metrics.silhouette_score(X_train, labels, metric='euclidean')

import os
try:
    os.remove(output_filename)
    os.remove('c.csv')
except OSError:
    pass

raw_data = {'PD': tokens,
        'label': kmeans.labels_}

raw_data2 = {'cluster name':terms}
df = pd.DataFrame(raw_data, columns = ['PD', 'label'])
df.to_csv(output_filename)
df = pd.DataFrame(raw_data2, columns = ['cluster name'])
df.to_csv('c.csv')
