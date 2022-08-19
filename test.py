from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

uda_pre = np.load('uda_pre.npy')
uda_true = np.load('uda_true.npy')
uda_pool = np.load('uda_pool.npy')

distill_pre = np.load('distill_pre.npy')
distill_true = np.load('distill_true.npy')
distill_pool = np.load('distill_pool.npy')


tsne = TSNE(n_components=2, init="pca", random_state=0)
all_pool = np.concatenate((uda_pool, distill_pool))
print(all_pool.shape)
all_tsne = tsne.fit_transform(all_pool)
# uda_tsne=tsne.fit_transform(uda_pool)
# distill_tsne=tsne.fit_transform(distill_pool)

uda_tsne = all_tsne[:5000]
distill_tsne = all_tsne[5000:]
print(distill_tsne.shape)
print(uda_tsne.shape)

plt.figure(figsize=(16, 16))
plt.scatter(uda_tsne[:, 0][uda_pre == 1], uda_tsne[:, 1][uda_pre == 1], c='r')
plt.scatter(uda_tsne[:, 0][uda_pre != 1], uda_tsne[:, 1][uda_pre != 1], c='b')

plt.scatter(distill_tsne[:, 0][distill_pre == 1],
            distill_tsne[:, 1][distill_pre == 1], c='y')
plt.scatter(distill_tsne[:, 0][distill_pre != 1],
            distill_tsne[:, 1][distill_pre != 1], c='g')


pca = PCA(n_components=2)
pca_all = pca.fit_transform(all_pool)
uda_pca = pca_all[:5000]
distill_pca = pca_all[5000:]

plt.figure(figsize=(16, 16))
plt.scatter(uda_pca[:, 0][uda_pre == 1], uda_pca[:, 1][uda_pre == 1], c='r')
plt.scatter(uda_pca[:, 0][uda_pre != 1], uda_pca[:, 1][uda_pre != 1], c='b')

plt.scatter(distill_pca[:, 0][distill_pre == 1],
            distill_pca[:, 1][distill_pre == 1], c='y')
plt.scatter(distill_pca[:, 0][distill_pre != 1],
            distill_pca[:, 1][distill_pre != 1], c='g')
