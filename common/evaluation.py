import numpy as np
import math
from scipy.special import comb

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def evaluate_cluster(features, labels, n_classes):
    # k-means algorithms
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(features)
    centers = kmeans.cluster_centers_

    # k-NN algorithms
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers, range(len(centers)))

    idx_feat = neigh.predict(features)
    nums = len(features)
    ds = np.zeros(nums)
    for i in range(nums):
        ds[i] = np.linalg.norm(features[i, :] - centers[idx_feat[i], :])

    labels_pre = np.zeros(nums)
    for i in range(n_classes):
        idx = np.where(idx_feat == i)[0]
        ind = np.argmin(ds[idx])
        cid = idx[ind]
        labels_pre[idx] = cid

    NMI, F1 = compute_cluster_metric(labels, labels_pre)
    return NMI, F1


def compute_cluster_metric(labels, labels_pre):
    N = len(labels)
    centers = np.unique(labels)
    n_clusters = len(centers)

    # count the number
    count_cluster = np.zeros(n_clusters)
    for i in range(n_clusters):
        count_cluster[i] = len(np.where(labels == centers[i])[0])

    # map labels_pre into item_map
    keys = np.unique(labels_pre)
    nums_item = len(keys)
    values = range(nums_item)
    item_map = dict()
    for i in range(nums_item):
        item_map[keys[i]] = values[i]
        # item_map.update([keys[i], values[i]])

    # count the number
    count_item = np.zeros(nums_item)
    for i in range(N):
        idx = item_map[labels_pre[i]]
        count_item[idx] += 1

    # compute purity
    purity = 0
    for i in range(n_clusters):
        member = np.where(labels == centers[i])[0]
        member_ids = labels_pre[member]

        count = np.zeros(nums_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)

    purity = purity / N

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((n_clusters, nums_item))
    for i in range(N):
        index_cluster = np.where(labels[i] == centers)[0]
        index_item = item_map[labels_pre[i]]
        count_cross[index_cluster, index_item] += 1

    I = 0
    for k in range(n_clusters):
        for j in range(nums_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s

    # entropy
    H_cluster = 0
    for k in range(n_clusters):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(nums_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)

    # computer F-measure
    tp_fp = 0
    for k in range(n_clusters):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(n_clusters):
        member = np.where(labels == centers[k])[0]
        member_ids = labels_pre[member]

        count = np.zeros(nums_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(nums_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(nums_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute True Negative (TN)
    tn = N * (N - 1) / 2 - tp - fp - fn

    # compute RI
    RI = (tp + tn) / (tp + fp + fn + tn)

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F1 = (beta * beta + 1) * P * R / (beta * beta * P + R)

    return NMI, F1


def evaluate_recall(features, labels):
    class_ids = labels
    dims = features.shape

    D2 = distance_matrix(features)

    # set diagonal to very high number
    num = dims[0]
    D = np.sqrt(np.abs(D2))
    diagn = np.diag([float('inf') for i in range(0, D.shape[0])])
    D = D + diagn
    recall = []
    for K in [1, 5]:
        recall.append(compute_recall_at_K(D, K, class_ids, num))

    return recall


def evaluate_recall_asym(features_gallery, labels_gallery, features_query, labels_query):
    dims = features_query.shape

    D2 = distance_matrix_asym(features_query, features_gallery)

    # set diagonal to very high number
    num = dims[0]
    D = np.sqrt(np.abs(D2))
    recall = []
    for K in [1, 10, 20, 30, 40]:
        recall.append(compute_recall_at_K_asym(D, K, labels_gallery, labels_query, num))

    return recall


# Symmetric distance computation (SDC)
def distance_matrix(X):
    X = np.matrix(X)
    m = X.shape[0]
    t = np.matrix(np.ones([m, 1]))
    x = np.matrix(np.empty([m, 1]))
    for i in range(0, m):
        n = np.linalg.norm(X[i, :])
        x[i] = n * n
    D = x * np.transpose(t) + t * np.transpose(x) - 2 * X * np.transpose(X)
    return D


# Asymmetric distance computation (ADC)
def distance_matrix_asym(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
    BT = B.transpose()
    vecProd = A * BT
    SqA = A.getA() ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B.getA() ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    ED = (SqED.getA())  # **0.5
    return np.matrix(ED)


def compute_recall_at_K(D, K, class_ids, num):
    num_correct = 0
    for i in range(0, num):
        this_gt_class_idx = class_ids[i]
        this_row = D[i, :]
        inds = np.array(np.argsort(this_row))[0]
        knn_inds = inds[0:K]
        knn_class_inds = [class_ids[i] for i in knn_inds]

        if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
            num_correct = num_correct + 1
    recall = float(num_correct) / float(num)

    return recall


def compute_recall_at_K_asym(D, K, class_ids_gallery, class_ids_query, num):
    num_correct = 0
    for i in range(0, num):
        this_gt_class_idx = class_ids_query[i]
        this_row = D[i, :]
        inds = np.array(np.argsort(this_row))[0]
        knn_inds = inds[0:K]
        knn_class_inds = [class_ids_gallery[i] for i in knn_inds]

        if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
            num_correct = num_correct + 1
    recall = float(num_correct) / float(num)

    return recall
