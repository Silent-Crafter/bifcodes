import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score

X = pd.read_csv('./CC GENERAL.csv')
X = X.drop('CUST_ID', axis=1)

X.ffill(inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_normalized = normalize(X_scaled)
X_normalized_df = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principle = pca.fit_transform(X_normalized_df)
X_principle = pd.DataFrame(X_principle)
X_principle.columns = ['P1', 'P2']

plt.figure(figsize =(10, 10))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principle, method ='ward')))
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,10))
rc = 0
cc = 0

agglomeratives = []
c_values = []

for i in range(2, 7):
    if cc >= 3:
        cc = 0
        rc += 1
    
    ac = AgglomerativeClustering(n_clusters = i)
    c = ac.fit_predict(X_principle)
    ax[rc][cc].scatter(X_principle['P1'], X_principle['P2'], 
                c = c, cmap ='rainbow')
    agglomeratives.append(ac)
    c_values.append(c)

    cc += 1

plt.show()
k = [2, 3, 4, 5, 6]
silhouette_scores = [
    silhouette_score(X_principle, c)
    for c in c_values
]

print(silhouette_scores)
plt.bar(k,silhouette_scores)
plt.xlabel('Number of Clusters', fontsize = 20)
plt.ylabel('S(i)',fontsize = 20)
plt.show()
