# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)
import psycopg2
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classy', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()

# Выбираем значения свойств (0 или 1) из таблицы dat
cursor.execute("SELECT features FROM dat_files;")

# Cохраняем результат в список, хранящий кортежи вида
# (["0" "0" "0" ... "0" "0" "1"],)
sites_features = cursor.fetchall()
# print (sites_features)

all_features = []  # хранит таблицу 5000 строк по 250 признаков
for site in sites_features:
    to_int = []
    for i in site[0]:
        if i.isdigit():
            to_int.append(int(i))
    site = (to_int)
    all_features.append(to_int)
# print all_features

# Формируем таблицы 5000х50 признаков для каждой категории
# То есть для категории есть только её признаки у каждого из 5К сайтов
adult_features = []
alcohol_features = []
ecommerce_features = []
medical_features = []
religion_features = []
for cur in all_features:
    adult_features.append(cur[:50])
    alcohol_features.append(cur[50:100])
    ecommerce_features.append(cur[100:150])
    medical_features.append(cur[150:200])
    religion_features.append(cur[200:])

# print ecommerce_features

# Выбираем категории сайтов из таблицы dat
cursor.execute("SELECT categ_basis FROM dat_files;")

# Сохраняем категорию каждого сайта в список (длиной 5000)
categ_vals = [row[0][20:] for row in cursor]

adult_categ = []
alcohol_categ = []
ecommerce_categ = []
medical_categ = []
religion_categ = []
for cat in categ_vals:
    if cat == "adult":
        adult_categ.append(cat)
    else:
        adult_categ.append("Not_adult")

    if cat == "alcohol":
        alcohol_categ.append(cat)
    else:
        alcohol_categ.append("Not_alcohol")

    if cat == "ecommerce":
        ecommerce_categ.append(cat)
    else:
        ecommerce_categ.append("Not_ecommerce")

    if cat == "medical":
        medical_categ.append(cat)
    else:
        medical_categ.append("Not_medical")

    if cat == "religion":
        religion_categ.append(cat)
    else:
        religion_categ.append("Not_religion")

print adult_categ
print alcohol_categ
print ecommerce_categ
print medical_categ
print religion_categ
