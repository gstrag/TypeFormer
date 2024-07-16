import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.test_config import test_configs

plt.gca().set_aspect('equal', adjustable='box')


embeddings = np.load(test_configs.results_dir + "test_embeddings_all_users.npy", allow_pickle=True).item()

user_embeddings = []
users_considered = [x for x in list(embeddings.keys())[:10]]
for user in users_considered:
    for session in list(embeddings[user].keys()):
        user_embeddings.append(embeddings[user][session])
user_embeddings = np.array(user_embeddings, dtype=float)
embeddings_to_sne_transformed = TSNE(n_components=2, perplexity=14, init='pca', n_iter=1000).fit_transform(user_embeddings)


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot()#projection='3d')

fontsizeaxis = 30
fontsizelegend = 22
markersize = 100
fontsizeticks = 25


ax.set_xlabel("t-SNE Dimension 1", fontsize=fontsizeaxis)
ax.set_ylabel("t-SNE Dimension 2", fontsize=fontsizeaxis)
for i in range(int(len(user_embeddings)/15)):
    ax.scatter(embeddings_to_sne_transformed[15*i:15*(i+1),0], embeddings_to_sne_transformed[15*i:15*(i+1),1], label='User ' + str(i), s = markersize)#, alpha = 1.0)
ax.grid()
ax.legend(loc = 'lower right', fontsize=fontsizelegend)
ax.tick_params(axis='both', labelsize=fontsizeticks)
plt.savefig('tsne.pdf')
