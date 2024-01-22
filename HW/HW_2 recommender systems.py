import numpy as np
import pandas as pd
import scipy.linalg

# Загрузка данных
# Предполагается, что у нас загружен датасет Netflix Prize Data в переменную 'df'.
df = pd.read_csv('/path/to/netflix-prize-data/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'])

# Преобразование данных в матрицу 'пользователь-фильм'
user_movie_matrix = df.pivot(index='User', columns='Movie', values='Rating').fillna(0).values

# SVD-разложение матрицы
U, s, V = scipy.linalg.svd(user_movie_matrix)

# Извлечение первых двух строк из матриц U и V
U_2 = U[:, :2]
V_2 = V[:2, :]

# Выбор пользователя №2
user_2 = user_movie_matrix[1, :]

# Вычисление представления пользователя сниженной размерности
low_dim = np.dot(user_2, V_2.T)

# Обратная трансформация в вектор оценок фильмов
inv_transform = np.dot(low_dim, V_2)

# Получение нового вектора оценок для пользователя №2
new_ratings_user_2 = inv_transform[:]

# Индекс непросмотренного фильма с наибольшей оценкой
index_highest_rated_unwatched = np.argmax(new_ratings_user_2 * (user_2 == 0))

# Вывод результата
print("Индекс непросмотренного фильма с наибольшей оценкой:", index_highest_rated_unwatched)

# SVD-разложение на новом пользователе
new_user = np.array((0, 0, 3, 4, 0))

# SVD-разложение на матрице user_movie_matrix без нового пользователя
U, s, V = scipy.linalg.svd(user_movie_matrix)

# Извлекаем первые три компоненты
U_3 = U[:, :3]
V_3 = V[:3, :]

# Производим вычисление представления нового пользователя
low_dim_new_user = np.dot(new_user, V_3.T)

# Вывод индекса непросмотренного новым пользователем фильма с наибольшей оценкой
index_highest_rated_unwatched_new_user = np.argmax(low_dim_new_user)

print("Индекс непросмотренного новым пользователем фильма с наибольшей оценкой:", index_highest_rated_unwatched_new_user)