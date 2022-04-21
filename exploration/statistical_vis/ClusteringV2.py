from gettext import npgettext
import re
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt
import umap
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.manifold import TSNE
import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_mutual_info_sco
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score



def scaling(df):
    columns = list(df.columns)
    for column in columns:
        df[str(column)] = df[str(column)].fillna(
            df[str(column)].mean())
    scaler = preprocessing.StandardScaler()
    column_names = df.columns
    fit = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(fit, columns=column_names)
    return scaled_df


def UMAP(df_labels, scaled_df):
    reducer = umap.UMAP()
    umap_df = reducer.fit_transform(scaled_df)
    # Plot
    scatter = ax1.scatter(umap_df[:, 0], umap_df[:, 1],
                          c=df_labels.cat.codes, cmap='Paired', label='inline label', s=8)
    legend1 = ax1.legend(*scatter.legend_elements(), loc='center left',
                         bbox_to_anchor=(1, 0.5), title="Sow Max Parity")
    ax1.add_artist(legend1)
    ax1.set_title("Data projected by UMAP")
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    return umap_df


def pca(df_labels, scaled_df):
    pca = PCA(n_components=2)
    pca.fit(scaled_df)
    pca_df = pca.transform(scaled_df)
    print("The varience encoded by this PCA projection is:")
    print(pca.explained_variance_)
    # Plot
    scatter = ax1.scatter(pca_df[:, 0], pca_df[:, 1],
                          c=df_labels.cat.codes, cmap='Paired', s=8)
    legend1 = ax1.legend(*scatter.legend_elements(), loc='center left',
                         bbox_to_anchor=(1, 0.5), title="Sow Max Parity")
    ax1.add_artist(legend1)
    ax1.set_title("Data reduced by PCA")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    return pca_df


def tsne(df_labels, scaled_df):
    tsne_df = TSNE(n_components=2, perplexity=10, learning_rate='auto',
                   init='random').fit_transform(scaled_df)
    # Plot
    scatter = ax1.scatter(
        tsne_df[:, 0], tsne_df[:, 1], c=df_labels.cat.codes, cmap='Paired', s=8)
    legend1 = ax1.legend(*scatter.legend_elements(), loc='center left',
                         bbox_to_anchor=(1, 0.5), title="Sow Max Parity")
    ax1.add_artist(legend1)
    ax1.set_title("Data reduced by t-SNE")
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    return tsne_df


def kmeans(reduced_df):
    kmeans = KMeans(n_clusters=5, random_state=0)
    y_pred_Kmeans = kmeans.fit_predict(reduced_df)
    scatter = ax2.scatter(reduced_df[:, 0], reduced_df[:, 1],
                          c=y_pred_Kmeans, cmap='Paired', label='inline label', s=8)
    legend3 = ax2.legend(*scatter.legend_elements(),
                         loc='center left', bbox_to_anchor=(1, 0.5), title="Class")
    ax2.set_title("Data clustered by K-Means")
    ax2.add_artist(legend3)
   # print(nmi(reduced_df,y_pred_Kmeans))
    km_shil = silhouette_score(reduced_df, kmeans.labels_, metric='euclidean')
    print(km_shil)
    plt.xlabel('Z1')
    plt.ylabel('Z1')

    '''Export cluster values to file'''
    cluster_map = pd.DataFrame()
    cluster_map['Max_Parity'] = df['max_parity'].values
    cluster_map['total_weaned'] = df['total_weaned'].values
    cluster_map['mean_bbp'] = df['mean_bbp'].values
    cluster_map['age_at_death'] = df['age_at_death'].values
    #cluster_map['weaned_p1'] = df['weaned_p1'].values
    #cluster_map['weight_at_first_weaning'] = df['weight_at_first_weaning'].values
    # cluster_map['first_lactation_weight_loss'] = df['first_lactation_weight_loss'].values #DLpeak values index + 2 = row num in csv file due to indexing starting at 0
    # cluster_map['data_Score'] = df['Score'].values
    #cluster_map['first_lact_weight_loss'] = df['first_lactation_weight_loss'].values
    # cluster_map['cluster'] = kmeans.labels_
    cluster_map['cluster'] = kmeans.labels_
    for cluster in range (kmeans.n_clusters):
        kmeans_data = cluster_map[cluster_map.cluster == cluster]
        kmeans_data.to_csv('tsne_kmeans_4feature.csv', index=False, mode='a')
    return y_pred_Kmeans


def dbscan(reduced_df):
    dbscan = DBSCAN(eps=1.45, min_samples=4)
    dbscan.fit(reduced_df)
    y_pred_dbscan = dbscan.fit_predict(reduced_df)
    scatter = ax3.scatter(reduced_df[:, 0], reduced_df[:, 1],
                          c=y_pred_dbscan, cmap='Paired', label='inline label', s=8)
    legend2 = ax3.legend(*scatter.legend_elements(),
                         loc='center left', bbox_to_anchor=(1, 0.5), title="Class")
    ax3.add_artist(legend2)
    ax3.set_title("Data clustered by DBSCAN")
    # print(nmi(reduced_df,y_pred_dbscan))
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    db_shil = silhouette_score(reduced_df, dbscan.labels_, metric='euclidean')
    print(db_shil)
    return y_pred_dbscan


def spectral(reduced_df):
    s_cluster = SpectralClustering(
        n_clusters=5, eigen_solver='arpack', affinity="nearest_neighbors")
    spectral_y_pred = s_cluster.fit_predict(reduced_df)
    scatter = ax4.scatter(reduced_df[:, 0], reduced_df[:, 1],
                          c=spectral_y_pred, s=8, cmap='Paired', label='inline label')
    legend4 = ax4.legend(*scatter.legend_elements(),
                         loc='center left', bbox_to_anchor=(1, 0.5), title="Class")
    ax4.add_artist(legend4)
    ax4.set_title("Data clustered by Spectral Clustering")
    # print(nmi(reduced_df,spectral_y_pred))
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    cluster_map = pd.DataFrame()
    cluster_map['Max_Parity'] = df['max_parity'].values
    cluster_map['total_weaned'] = df['total_weaned'].values
    cluster_map['mean_bbp'] = df['mean_bbp'].values
    #cluster_map['weaned_p1'] = df['weaned_p1'].values
    #cluster_map['weight_at_first_weaning'] = df['weight_at_first_weaning'].values
    # cluster_map['first_lactation_weight_loss'] = df['first_lactation_weight_loss'].values #DLpeak values index + 2 = row num in csv file due to indexing starting at 0
    # cluster_map['data_Score'] = df['Score'].values
    #cluster_map['first_lact_weight_loss'] = df['first_lactation_weight_loss'].values
    # cluster_map['cluster'] = s_cluster.labels_
    # cluster_map['cluster'] = s_cluster.labels_
    # for cluster in range (4):
    #     kmeans_data = cluster_map[cluster_map.cluster == cluster]
    #     kmeans_data.to_csv('pca_sclus_3feature.csv', index=False, mode='a')
    spec_shil = silhouette_score(reduced_df, s_cluster.labels_, metric='euclidean')
    print(spec_shil)
    return spectral_y_pred



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig, (ax1) = plt.subplots(1, 1)
# fig, (ax3) = plt.subplots(1, 1)
df = pd.read_csv('Data/pigs3csv.csv')

# bbp = df.groupby(['PIG_ID']).apply(lambda x: (x["Age at farrowing"] - x["Age at farrowing"].shift(1)))
# print("bbp:", bbp[:50])
# bbpmean = bbp.groupby(['PIG_ID']).mean()
# print("bppmean:", bbpmean[:50])
# print("loc:", bbpmean[unique_ids[5]])
# df['Mean BBP'] = np.NaN

df_labels = df['max_parity'].astype('category')
# df = df[['Parity', 'Age at Death', 'Gilt Insemination Weight (kg)', 'Backfat (mm)', 'Age at first insemination', 'Age at farrowing', 'Farrowing Weight (kg)', 'Backfat at Farrowing (mm)', 'Age at Weaning', 'Weaning Weight (kg)', 'Backfat at Weaning (mm)', 'Total Number Born', 'Number Born Alive', 'Number Born Dead', 'Mummified Piglets', 'Number to be weaned', 'Pigs Weaned',
#         'Lactation Length', 'Gestation Length']]
#df = df[['mean_bbp', 'total_weaned', 'max_parity', 'weaned_p1', 'weight_at_first_weaning', 'first_lactation_weight_loss']]
df = df[['mean_bbp', 'total_weaned', 'max_parity','age_at_death']]
# relates colours back to technology (colors in scatter points by 'technology or other parameter)
scaled_df = scaling(df)
#pca_df = pca(df_labels, scaled_df)
tsne_df = tsne(df_labels, scaled_df)
#umap_df = UMAP(df_labels, scaled_df)
#pickle.dump(tsne_df, open('tsne_proj', 'wb'))
#tsne_df = pickle.load(open('tsne_proj', 'rb'))


kmeans = kmeans(tsne_df)
dbscan = dbscan(tsne_df)
spectral = spectral(tsne_df)


# # Export cluster values to file
# cluster_map = pd.DataFrame()
# cluster_map['data_Parity'] = df['max_parity'].values
# cluster_map['data_Page_at_death'] = df['age_at_death'].values
# cluster_map['mean_bbb'] = df['mean_bbb'].values
# #cluster_map['data_first_insem_age'] = df['age_at_first_insemination'].values
# #cluster_map['first_wean_weight'] = df['weight_at_first_weaning'].values
# # cluster_map['data_Technology'] = df['Technology'].values #DLpeak values index + 2 = row num in csv file due to indexing starting at 0
# # cluster_map['data_Score'] = df['Score'].values
# #cluster_map['first_lact_weight_loss'] = df['first_lactation_weight_loss'].values
# # cluster_map['cluster'] = kmeans.labels_
# cluster_map['cluster'] = kmeans.labels_
# for cluster in range (kmeans.n_clusters):
#      kmeans_data = cluster_map[cluster_map.cluster == cluster]
#      kmeans_data.to_csv('kmeans_tsne.csv', index=False, mode='a')

plt.show()


'''elbow plot'''
# WCSS = []
# for i in range(1,11):
#     model = KMeans(n_clusters = i,init = 'k-means++')
#     model.fit(pca_df)
#     WCSS.append(model.inertia_)
# fig = plt.figure(figsize = (7,7))
# plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
# plt.xticks(np.arange(11))
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.show()

'''3d plotting'''
# kmeans = KMeans(n_clusters=5, random_state=0)
# kmeans_df = kmeans.fit_predict(scaled_df)
# print(np.shape(kmeans_df))
# fig = fig = px.scatter_3d(scaled_df, x='mean_bbp', y='total_weaned', z='max_parity',
#               color=kmeans_df)
# fig.show()

'''dbscan epsilon determination'''
# neighbors = NearestNeighbors(n_neighbors=4)
# neighbors_fit = neighbors.fit(tsne_df)
# distances, indices = neighbors_fit.kneighbors(tsne_df)
# distances = np.sort(distances, axis=0)
# distances = distances[:, 1]
# plt.plot(distances)
# plt.xlabel('Average K-distance')
# plt.ylabel('eps')
# plt.show()

'''PCA Cumulative Variance'''
# pca = PCA().fit(scaled_df)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()
