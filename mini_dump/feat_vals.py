# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)
import psycopg2
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pydotplus
import datetime
import xlsxwriter
from sklearn.metrics import precision_recall_fscore_support as score

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classification', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()

# Выбираем названия признаков
cursor.execute("SELECT feature_val FROM aml_files;")

# Сохраняем названия признаков в список (длиной 250)
feature_vals = [row[0] for row in cursor]
# print(feature_vals)

# Выбираем id сайтов и значения свойств (0 или 1) из таблицы dat
cursor.execute("SELECT website_id,features FROM dat_files;")

# Cохраняем результат в список, хранящий кортежи вида
# (3241859L, '"0" "0" "0" ... "0" "0" "1" ')
sites_features = cursor.fetchall()

X = []
# Преобразуем в список кортежей с массивом чисел
# вместо строки (3241859L, [0, 0, 0, ... , 1])
for site in sites_features:
    to_int = []
    for i in site[1]:
        if i.isdigit():
            to_int.append(int(i))
    site = (site[0], to_int)
    X.append(to_int)
# print(sites_features)

# Выбираем категории сайтов из таблицы dat
cursor.execute("SELECT categ_basis FROM dat_files;")

# Сохраняем категорию каждого сайта в список (длиной 5000)
categ_vals = [row[0][20:] for row in cursor]
y = categ_vals
# print(categ_vals)

################################################################

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

# В качестве классификатора выбрано Дерево решений
clf = tree.DecisionTreeClassifier(min_samples_leaf=30)
# Обучение классификатор
clf = clf.fit(X_train, y_train)

# Вероятностная классификация
pred_prob = clf.predict_proba(X_test)

# Таблица вероятностей принадлежности к категориям
"""for item in pred_prob:
    print item"""

# Категории с учётом неизвестной
class_labels = ['adult', 'alcohol', 'ecommerce',
                'medical', 'religion', 'Unknown']
y_pred_with_prob = []

# Цикл по таблице вероятностей (размером 1000)
# По умолчанию устанавливается категория Unknown
# Если одна из вероятностей cur_prob > x, записываем соответствующую ей
# категорию в результат
for probs in pred_prob:
    y_pred_with_prob.append(class_labels[5])
    for i, cur_prob in enumerate(probs):
        if cur_prob > 0.8:
            y_pred_with_prob[-1] = class_labels[i]

# Формирование матрицы и вывод в консоль
orig_conf_mat = confusion_matrix(y_test, y_pred_with_prob, labels=class_labels)
report = classification_report(y_test, y_pred_with_prob, labels=class_labels)
print "\nConfusion matrix (with probability):\n\n", orig_conf_mat
print "\n", report

# Преобразование (транспонирование) матрицы для отображения в отчёте excel
conf_mat = map(list, zip(*orig_conf_mat))

corr_pred = 0  # Количество верно классифицированных сайтов
unknown = 0  # Количество сайтов, отнесенных к категории неизвестно
total_sites = len(y_pred_with_prob)  # Всего сайтов изначально

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
workbook = xlsxwriter.Workbook('Example1.xlsx')  # Имя файла
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
precision, recall, fscore, support = score(y_test, y_pred_with_prob)
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

# Графическое представление построенного дерева
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(
    "tree_" + datetime.datetime.now().strftime("%H:%M:%S") + ".pdf")
