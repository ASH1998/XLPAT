# ! pip install git+https://github.com/LIAAD/yake

import pandas as pd
from tqdm import tqdm
import _pickle as cPickle
import yake

from utils_brownClustering import *
from nltk.tokenize import RegexpTokenizer

df = pd.read_csv('Dataset.csv')

#semi supervised keyword extractor
kwextractor = yake.KeywordExtractor(lan='en',
    n=3,
    dedupLim=0.9,
    dedupFunc='seqm',
    windowsSize=1,
    top=3)

# create empty list to save our "problems" to
problems = []

# subsample forum problems
sample_text = df.problems.astype(str)

for text in tqdm(sample_text):
    text_keywords = kwextractor.extract_keywords(text)
    
    sentence_output = ""
    for word, number in text_keywords:
        sentence_output += word + ","
    problems.append(sentence_output)

# tokenization of keywords
tokenizer = RegexpTokenizer(r'\w+')
sample_data_tokenized = [w.lower() for w in problems]
sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]

# training of corpus
corpus = Corpus(sample_data_tokenized, 0.001)
clustering = BrownClustering(corpus, 3)
clustering.train()


# clustering.get_similar('magnet')

# so it looks like all the clustres have been concatenated to a single array
# but they're in alphabetical order so we can use that to un-cat them 
megacluster = clustering.helper.get_cluster(0)

# create list with one sub list
cluster_list = [[]] 
list_index = 0

# look at all words but last (since we compare each word
# to the next word)
for i in range(len(megacluster) - 1):
    if megacluster[i - 1] < megacluster[i]:
        cluster_list[list_index].append(megacluster[i])
    else:
        cluster_list.append([])
        list_index = list_index + 1
        
        cluster_list[list_index].append(megacluster[i])

clusterdf = pd.DataFrame([cluster_list])
clusterdf = clusterdf.transpose()
clusterdf.columns = ['clsuter']
clusterdf.to_csv('cluster.csv')


df['kw'] = problems

clusters = []
for i in tqdm(df.index):
    kw_s = df.kw[i].replace(',', ' ')
#     print(kw_s.split())
    clusters_j = []
    for j in clusterdf.index:
        if any(elem in kw_s.split() for elem in clusterdf['clsuter'][j]):
#             print(clusterdf['clsuter'][j], j)
#             print(kw_s.split())
            clusters_j.append(j)
        
    clusters.append(clusters_j)

df['clusters'] = clusters

df.to_csv('clustered_Dataset.csv')

with open(r"clusteringObject.pickle", "wb") as output_file:
    cPickle.dump(clustering, output_file)
