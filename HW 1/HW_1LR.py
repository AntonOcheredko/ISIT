from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Набор данных
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=(0.15, 0.85),
    class_sep=6.0,
    hypercube=False,
    random_state=2
)

# Разделение выборки на обучающую и проверочную
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Логистическая регрессия
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_predictions = log_reg.predict(X_test)

# k ближайшие соседи
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

# Доля верных ответов
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Матрица ошибок
log_reg_conf_matrix = confusion_matrix(y_test, log_reg_predictions)
knn_conf_matrix = confusion_matrix(y_test, knn_predictions)

# Точность, полнота и F-мера
log_reg_precision = precision_score(y_test, log_reg_predictions)
knn_precision = precision_score(y_test, knn_predictions)

log_reg_recall = recall_score(y_test, log_reg_predictions)
knn_recall = recall_score(y_test, knn_predictions)

log_reg_f1_score = f1_score(y_test, log_reg_predictions)
knn_f1_score = f1_score(y_test, knn_predictions)

# PR
log_reg_precision, log_reg_recall, _ = precision_recall_curve(y_test, log_reg_predictions)
knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn_predictions)

# ROC
log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test, log_reg_predictions)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_predictions)

# Вычисление Average Precision и ROC-AUC
log_reg_avg_precision = auc(log_reg_recall, log_reg_precision)
knn_avg_precision = auc(knn_recall, knn_precision)

log_reg_roc_auc = auc(log_reg_fpr, log_reg_tpr)
knn_roc_auc = auc(knn_fpr, knn_tpr)

# Графики
plt.figure(figsize=(10, 5))

# Распределение выборки по классам
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# PR-кривые
plt.subplot(122)
plt.plot(log_reg_recall, log_reg_precision, label=f"Logistic Regression (Avg Precision = {log_reg_avg_precision:.2f})")
plt.plot(knn_recall, knn_precision, label=f"k-Nearest Neighbors (Avg Precision = {knn_avg_precision:.2f})")
plt.legend()
plt.title("PR Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.tight_layout()
plt.show()