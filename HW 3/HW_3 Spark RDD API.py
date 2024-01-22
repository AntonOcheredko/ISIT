import csv
import requests
from io import StringIO
from geopy.distance import geodesic

# Функция для вычисления расстояния между двумя точками
def calculate_distance(lat1, lng1, lat2, lng2):
    coords_1 = (lat1, lng1)
    coords_2 = (lat2, lng2)
    return geodesic(coords_1, coords_2).km

# Заданные координаты
target_lat, target_lng = 55.751244, 37.618423

# Загрузка данных из CSV-файла по URL
csv_url = "https://raw.githubusercontent.com/SergUSProject/IntelligentSystemsAndTechnologies/main/HomeWork/spark/data/places.csv"
response = requests.get(csv_url)
csv_data = StringIO(response.text)
reader = csv.reader(csv_data)
header = next(reader)  # Чтение заголовка
places_data = list(reader)

# Используем индексы для доступа к данным
lat_index = header.index('55.77302899555391')
lng_index = header.index('37.678681276899106')

# Рассчитываем расстояние от заданной точки до каждого заведения и добавляем в список
distances = []
for place in places_data:
    lat = float(place[lat_index])
    lng = float(place[lng_index])
    distance = calculate_distance(target_lat, target_lng, lat, lng)
    distances.append((place[1], distance))  # Индекс 1 соответствует имени заведения

# Сортируем по расстоянию и выводим первые 10
distances.sort(key=lambda x: x[1])
print("Топ-10 наиболее близких заведений:")
for i in range(10):
    print(f"{distances[i][0]} - Расстояние: {distances[i][1]:.2f} км")

print("\nТоп-10 наиболее отдаленных заведений:")
# Сортируем в обратном порядке и выводим первые 10
distances.sort(key=lambda x: x[1], reverse=True)
for i in range(10):
    print(f"{distances[i][0]} - Расстояние: {distances[i][1]:.2f} км")