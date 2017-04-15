# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)
import psycopg2
from itertools import *
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xlsxwriter
from sklearn.metrics import precision_recall_fscore_support as score

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

# Подготовленные тестовые данные подаём на вход нужным
# деревьям и сохраняем результат работы каждого из деревьев
for itree, test, pred in izip(tree_list, test_list, pred_list):
    # Вероятностная классификация (причём внутри вер-сти [Not_cat, cat])
    predicted = itree.predict_proba(test)
    # print predicted
    # Заменяем содержимое списка на предсказанные значения
    pred[0:-1] = predicted

# print y_test[:3]
# print y_test[-3:]

# Категории с учётом неизвестной
class_labels = ['adult', 'alcohol', 'ecommerce',
                'medical', 'religion', 'Unknown']
y_pred = []

# Минимальное значение вероятности для отнесения к одной из категорий
probab = 0.3
# Количество спорных/непонятных ситуаций (Unknown)
bad = 0
# Цикл для определения окончательной категории каждого из тестовых экземпляров
for ad, al, ec, med, rel in izip(adult_pred, alco_pred, ecomrc_pred, med_pred, relig_pred):
    # Среди вероятностий принадлежности к категориям ищем максимум
    is_cat = [ad[1], al[1], ec[1], med[1], rel[1]]
    maxi = max(is_cat)
    # Если максимумов несколько или он меньше порогового значения,
    # итоговая категория данного экземпляра - Unknown
    if (is_cat.count(maxi) != 1)or(maxi < probab):
        bad += 1
        # print bad, "Problems...", maxi, is_cat.count(maxi)
        y_pred.append(class_labels[5])
    else:
        y_pred.append(class_labels[is_cat.index(maxi)])

# print y_pred

# Формирование матрицы и вывод в консоль
orig_conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
report = classification_report(y_test, y_pred, labels=class_labels)
print "\nConfusion matrix (with probability):\n\n", orig_conf_mat
print "\n", report

# Преобразование (транспонирование) матрицы для отображения в отчёте excel
conf_mat = map(list, zip(*orig_conf_mat))

corr_pred = 0  # Количество верно классифицированных сайтов
unknown = 0  # Количество сайтов, отнесенных к категории неизвестно
total_sites = len(y_pred)  # Всего сайтов изначально

# Цикл подсчёта верно распознанных и неизвестных сайтов
for i, cur_row in enumerate(conf_mat):
    corr_pred += cur_row[i]
    unknown += conf_mat[5][i]

errors = total_sites - corr_pred - unknown  # Количество ошибок
accur = corr_pred / float(total_sites)  # Общая точность

# Вывод результатов в консоль
print 'Total correct predicted:', corr_pred, ' (', accur, ')'
print 'Total Unknowns:', unknown, ' (', unknown / float(total_sites), ')'
print '\nAccuracy without unknowns:', corr_pred / float(total_sites - unknown)
print 'Errors without unknowns', (errors) / float(total_sites - unknown)

################################################################

# Создание отчёта в Excel
workbook = xlsxwriter.Workbook('report (min_leaf = ' + str(
    clf.min_samples_leaf) + '; prob = ' + str(probab) + ')' + '.xlsx')  # Имя файла
worksheet = workbook.add_worksheet()
# Добавление таблицы conf_mat в отчёт
worksheet.add_table('B2:G8', {'data': conf_mat, 'autofilter': False})
# Задание имён столбцов
class_labels_true = ['true adult', 'true alcohol', 'true ecommerce',
                     'true medical', 'true religion', 'true Unknown']
worksheet.write_row('B2', class_labels_true)
# Задание имён строк
class_labels_pred = ['pred. adult', 'pred. alcohol', 'pred. ecommerce',
                     'pred. medical', 'pred. religion', 'pred. Unknown']
worksheet.write_column('A3', class_labels_pred)

# Задание имён строк и столбцов таблицы метрик
worksheet.write_row('A10', ['category', 'precision',
                            'recall', 'F-measure', 'allSites'])
worksheet.write_column('A11', class_labels)
# Вычисление метрик
precision, recall, fscore, support = score(y_test, y_pred)
# Запись вычисленных метрик в таблицу (на 0-ой позиции Unknown)
for i, j in zip(range(1, 6), ['B11', 'B12', 'B13', 'B14', 'B15', 'B16']):
    worksheet.write_row(j, [precision[i], recall[i], fscore[i], support[i]])
worksheet.write_row('B16', [precision[0], recall[0], fscore[0], support[0]])

# Количество ошибок; всего сайтов; верно распознанных сайтов
worksheet.write_row('A18', ['totalErrPredSites',
                            'totalSitesCount', 'totalCorrPred'])
worksheet.write_row('A19', [errors, total_sites, corr_pred])

# Средние показатели точности и полноты
worksheet.write_row('A21', ['accuracy', 'averPrecision', 'averRecall'])
worksheet.write_row('A22', [accur, sum(
    precision) / float(len(precision) - 1), sum(recall) / float(len(recall) - 1)])

# Общее количество нераспознанных сайтов и ошибок
worksheet.write_row(
    'A24', ['totalUnknowns', 'totalUnrecognized', 'totalErrWithoutUnknown'])
worksheet.write_row('A25', [unknown, total_sites - corr_pred, errors])

# Подведение итогов: показатели в виде части от общего количества сайтов
worksheet.write('A28', 'SUMMARY')
worksheet.write_row('A29', ['Accuracy', 'Errors', 'Unknowns'])
worksheet.write_row(
    'A30', [accur, errors / float(total_sites), unknown / float(total_sites)])

worksheet.write_column(
    'A32', ['Accuracy without unknowns', corr_pred / float(total_sites - unknown)])
worksheet.write_column('A35', ['Errors without unknowns',
                               errors / float(total_sites - unknown)])

workbook.close()
