# ! pip install git+https://github.com/LIAAD/yake
# !pip install sentence-transformers
# !conda install hdbscan
# !pip install umap-learn

import pandas as pd
from tqdm import tqdm
import _pickle as cPickle
import matplotlib.pyplot as plt

import yake

import hdbscan
import umap


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

df = pd.read_csv('Dataset.csv')


#semi supervised keyword extractor
kwextractor = yake.KeywordExtractor(lan='en',
    n=3,
    dedupLim=0.9,
    dedupFunc='seqm',
    windowsSize=1,
    top=10)

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

embeddings = model.encode(problems, show_progress_bar=True)

# dimentionality reduction
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=2, 
                            metric='cosine').fit_transform(embeddings)

#cluster training
cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)


# visuals
# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(15, 5))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=5.5)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=5.5, cmap='hsv_r')
plt.colorbar()


df['cluster'] = cluster.labels_
df.to_csv('hdbscan_Dataset.csv')

#save pickle
with open(r"HDBSCANObject.pickle", "wb") as output_file:
    cPickle.dump(cluster, output_file)