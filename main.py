import random
import math


# Đọc dữ liệu từ file
def load_data(filepath):
    data = []
    labels = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                *features, label = line.strip().split(',')
                data.append([float(x) for x in features])
                labels.append(label)
    return data, labels


# Khởi tạo centroids ngẫu nhiên
def initialize_centroids(data, k):
    return random.sample(data, k)


# Tính khoảng cách Euclidean
def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5


# Phân loại dữ liệu vào các cụm
def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters


# Cập nhật centroids
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Kiểm tra cụm không rỗng
            new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append([0] * len(clusters[0][0]))  # Nếu cụm rỗng, gán centroid về 0
    return new_centroids


# Thuật toán K-means
def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids


# Tính F1-Score
def f1_score(true_labels, predicted_labels):
    label_clusters = {}
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if predicted_label not in label_clusters:
            label_clusters[predicted_label] = []
        label_clusters[predicted_label].append(true_label)

    f1_total = 0
    for label, cluster in label_clusters.items():
        tp = sum([1 for i in cluster if i == label])
        fp = len(cluster) - tp
        fn = true_labels.count(label) - tp
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_total += 2 * (precision * recall) / (precision + recall)

    return f1_total / len(label_clusters) if label_clusters else 0


# Tính RAND Index
def rand_index(true_labels, predicted_labels):
    tp_fp_tn_fn = len(true_labels) * (len(true_labels) - 1) / 2
    tp_fp = sum(
        1
        for i in range(len(true_labels))
        for j in range(i + 1, len(true_labels))
        if (true_labels[i] == true_labels[j]) == (predicted_labels[i] == predicted_labels[j])
    )
    return tp_fp / tp_fp_tn_fn if tp_fp_tn_fn > 0 else 0


# Tính NMI (Normalized Mutual Information)
def nmi(true_labels, predicted_labels):
    def entropy(labels):
        label_counts = {label: labels.count(label) for label in set(labels)}
        return -sum(count / len(labels) * math.log(count / len(labels), 2) for count in label_counts.values())

    true_entropy = entropy(true_labels)
    pred_entropy = entropy(predicted_labels)
    mutual_info = sum(
        true_labels.count(t) / len(true_labels) * math.log(true_labels.count(t) / len(predicted_labels), 2)
        for t in set(true_labels)
        if t in predicted_labels
    )
    return mutual_info / math.sqrt(true_entropy * pred_entropy) if true_entropy > 0 and pred_entropy > 0 else 0


# Tính DB Index (Davies-Bouldin Index)
def davies_bouldin_index(clusters, centroids):
    db_index = 0
    for i in range(len(clusters)):
        max_ratio = 0
        for j in range(len(clusters)):
            if i != j:
                dist = euclidean_distance(centroids[i], centroids[j])
                si = sum(euclidean_distance(point, centroids[i]) for point in clusters[i]) / len(clusters[i]) if clusters[i] else 0
                sj = sum(euclidean_distance(point, centroids[j]) for point in clusters[j]) / len(clusters[j]) if clusters[j] else 0
                ratio = (si + sj) / dist if dist > 0 else 0
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    db_index /= len(clusters) if clusters else 1
    return db_index


# Chạy và Đánh giá
if __name__ == "__main__":
    # Đường dẫn tới file dữ liệu IRIS
    filepath = r"D:\3-DAI_HOC\5_HocKy2\2_XLA_TGMT\2_Bai_tap\BT_30_10\pythonProject\BT_1_11\iris\iris.data"

    # Đọc dữ liệu
    data, true_labels = load_data(filepath)

    # Chạy K-means với số cụm k = 3
    k = 3
    clusters, centroids = kmeans(data, k)

    # Giả sử mỗi cụm là một nhãn (thay thế dựa trên ánh xạ cụm của bạn)
    predicted_labels = [i for i, cluster in enumerate(clusters) for _ in cluster]

    # Tính các chỉ số đánh giá
    f1 = f1_score(true_labels, predicted_labels)
    rand = rand_index(true_labels, predicted_labels)
    nmi_score = nmi(true_labels, predicted_labels)
    db_index = davies_bouldin_index(clusters, centroids)

    # In kết quả
    print(f"F1 Score: {f1}")
    print(f"RAND Index: {rand}")
    print(f"NMI: {nmi_score}")
    print(f"Davies-Bouldin Index: {db_index}")
