# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)
import psycopg2
from itertools import *
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

# Выбираем категории сайтов из таблицы dat
cursor.execute("SELECT categ_basis FROM dat_files;")

# Сохраняем категорию каждого сайта в список (длиной 5000)
categ_vals = [row[0][20:] for row in cursor]

#################################################
# Разбиваем всю выборку на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    all_features, categ_vals, test_size=0.2, stratify=categ_vals)

# Формируем таблицы 4000х50 признаков для каждой категории
# То есть для категории есть только её признаки у каждого из 4К сайтов
adult_feat, alco_feat, ecomrc_feat, med_feat, relig_feat = [], [], [], [], []
feat_list = [adult_feat, alco_feat, ecomrc_feat, med_feat, relig_feat]

for cur in X_train:
    adult_feat.append(cur[:50])
    alco_feat.append(cur[50:100])
    ecomrc_feat.append(cur[100:150])
    med_feat.append(cur[150:200])
    relig_feat.append(cur[200:])

# print ecomrc_feat

# Инициализация списков для хранения категорий
adult_categ, alco_categ, ecomrc_categ = [], [], []
med_categ, relig_categ = [], []
categ_list = [adult_categ, alco_categ, ecomrc_categ, med_categ, relig_categ]

# Для каждой категории определяем 2 класса
for cat in y_train:
    if cat == "adult":
        adult_categ.append(cat)
    else:
        adult_categ.append("Not_adult")

    if cat == "alcohol":
        alco_categ.append(cat)
    else:
        alco_categ.append("Not_alcohol")

    if cat == "ecommerce":
        ecomrc_categ.append(cat)
    else:
        ecomrc_categ.append("Not_ecommerce")

    if cat == "medical":
        med_categ.append(cat)
    else:
        med_categ.append("Not_medical")

    if cat == "religion":
        relig_categ.append(cat)
    else:
        relig_categ.append("Not_religion")

# print adult_categ, alco_categ, ecomrc_categ, med_categ, relig_categ

###############################################
# Обучение деревьев решений для каждой категории
# В качестве классификатора выбрано Дерево решений
clf = tree.DecisionTreeClassifier(min_samples_leaf=30)

# Обучение классификатора для каждой из категорий
adult_tree = clf.fit(adult_feat, adult_categ)
alco_tree = clf.fit(alco_feat, alco_categ)
ecomrc_tree = clf.fit(ecomrc_feat, ecomrc_categ)
med_tree = clf.fit(med_feat, med_categ)
relig_tree = clf.fit(relig_feat, relig_categ)
# Список всех деревьев
tree_list = [adult_tree, alco_tree, ecomrc_tree, med_tree, relig_tree]

# Подготовка тестовой выборки X_test для эксперимента
# Все списки хранят свои признаки для подачи на вход деревьям
adult_test, alco_test, ecomrc_test, med_test, relig_test = [], [], [], [], []
test_list = [adult_test, alco_test, ecomrc_test, med_test, relig_test]

for cur in X_test:
    adult_test.append(cur[:50])
    alco_test.append(cur[50:100])
    ecomrc_test.append(cur[100:150])
    med_test.append(cur[150:200])
    relig_test.append(cur[200:])

# Эти переменные хранят двумерные списки вероятностей
adult_pred, alco_pred, ecomrc_pred = [], [], []
med_pred, relig_pred = [], []
pred_list = [adult_pred, alco_pred, ecomrc_pred, med_pred, relig_pred]

for itree, test, pred in zip(tree_list, test_list, pred_list):
    # Вероятностная классификация (причём внутри вер-сти [Not_cat, cat])
    predicted = itree.predict_proba(test)
    print predicted
    # Заменяем содержимое списка на предсказанные значения
    pred[0:-1] = predicted

print y_test[:3]
print y_test[-3:]

class_labels = ['adult', 'alcohol', 'ecommerce',
                'medical', 'religion', 'Unknown']
y_pred = []

bad = 0
for ad, al, ec, med, rel in zip(adult_pred, alco_pred, ecomrc_pred, med_pred, relig_pred):
    is_cat = [ad[1], al[1], ec[1], med[1], rel[1]]
    maxi = max(is_cat)
    if (is_cat.count(maxi) != 1):
        bad += 1
        # print bad, "FUCKFUCKFUCK", maxi, is_cat.count(maxi)
        y_pred.append(class_labels[5])
    else:
        y_pred.append(class_labels[is_cat.index(maxi)])

# print y_pred

orig_conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
report = classification_report(y_test, y_pred, labels=class_labels)
print "\nConfusion matrix (with probability):\n\n", orig_conf_mat
print "\n", report
