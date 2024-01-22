# Загружаем библиотеки
from numpy import savetxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import requests

# Получаем файл `train.csv` из `GitHub`
url = "https://raw.githubusercontent.com/SergUSProject/IntelligentSystemsAndTechnologies/main/Practice/datasets/train.csv"
response = requests.get(url)

with open("train.csv", "wb") as file:
    file.write(response.content)

# Далее делим набор данных на обучающие и тестовые (70% и 30%) и загружаем обучающие данные из файла `train.csv` в дамп `training_set.pkl` (чтобы не хранить в оперативной памяти)
data = pd.read_csv('train.csv', low_memory=False)
target_column = data.columns[0]
feature_columns = data.columns[1:]

train_data, test_data = train_test_split(data, test_size=0.3)
train_data.to_pickle('training_set.pkl')
train_data = pd.read_pickle('training_set.pkl')

# Из обучающей части выделяем целевую переменную и остальную часть
target_tr = train_data[target_column].astype('str')
train_tr = train_data[feature_columns].astype('str')

# Применяем LabelEncoder для преобразования строковых значений в числа
label_encoder = LabelEncoder()
target_tr_encoded = label_encoder.fit_transform(target_tr)
train_tr_encoded = train_tr.apply(label_encoder.fit_transform)

# Сохраняем тестовую выборку (без целевой переменной) в файл test_set.pkl
test_without_target = test_data[feature_columns].astype('str')
test_without_target.to_pickle('test_set.pkl')
test = pd.read_pickle('test_set.pkl')

# Обучаем дерево принятия решений
tree = DecisionTreeClassifier()
tree.fit(train_tr_encoded, target_tr_encoded)

# Записываем результат классификации на тестовом наборе, сохраненном в виде дампа test_set.pkl, в файл answer.csv
tree_predictions_encoded = tree.predict(test)
tree_predictions = label_encoder.inverse_transform(tree_predictions_encoded)
savetxt('answer.csv', tree_predictions, delimiter=',', fmt='%s')

# Оцените качество классификации при помощи метрики accuracy_score
test_target = test_data[target_column].astype('str')
test_target_encoded = label_encoder.transform(test_target)
accuracy = accuracy_score(test_target_encoded, tree_predictions_encoded)
print("Accuracy:", accuracy)