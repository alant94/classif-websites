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
    database='Full_Dump', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()

# Выбираем значения свойств (0 или 1) из таблицы dat
cursor.execute("SELECT features FROM dat_files WHERE entry_id < 10001;")

# Cохраняем результат в список, хранящий кортежи вида
# (["0" "0" "0" ... "0" "0" "1"],)
sites_features = cursor.fetchall()
# print (sites_features)

all_features = []  # хранит таблицу 10000 строк по 500 признаков
for site in sites_features:
    to_int = []
    for i in site[0]:
        if i.isdigit():
            to_int.append(int(i))
    site = (to_int)
    all_features.append(to_int)
# print all_features

# Выбираем категории сайтов из таблицы dat
cursor.execute("SELECT categ_basis FROM dat_files WHERE entry_id < 10001;")

# Сохраняем категорию каждого сайта в список (длиной 10000)
categ_vals = [row[0][40:] for row in cursor]
# print categ_vals

#################################################
# Разбиваем всю выборку на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    all_features, categ_vals, test_size=0.2, stratify=categ_vals)

# Формируем таблицы 8000х50 признаков для каждой категории
# То есть для категории есть только её признаки у каждого из 8К сайтов
mus_feat, gam_feat, chat_feat, ecomrc_feat, adult_feat = [], [], [], [], []
alco_feat, hunt_feat, news_feat, med_feat, relig_feat = [], [], [], [], []
feat_list = [mus_feat, gam_feat, chat_feat, ecomrc_feat,
             adult_feat, alco_feat, hunt_feat, news_feat, med_feat, relig_feat]

for cur in X_train:
    mus_feat.append(cur[:50])
    gam_feat.append(cur[50:100])
    chat_feat.append(cur[100:150])
    ecomrc_feat.append(cur[150:200])
    adult_feat.append(cur[200:250])
    alco_feat.append(cur[250:300])
    hunt_feat.append(cur[300:350])
    news_feat.append(cur[350:400])
    med_feat.append(cur[400:450])
    relig_feat.append(cur[450:])
# print ecomrc_feat

# Инициализация списков для хранения категорий
mus_categ, gam_categ, chat_categ, ecomrc_categ, adult_categ = [], [], [], [], []
alco_categ, hunt_categ, news_categ, med_categ, relig_categ = [], [], [], [], []
categ_list = [mus_categ, gam_categ, chat_categ, ecomrc_categ, adult_categ,
              alco_categ, hunt_categ, news_categ, med_categ, relig_categ]

classes_list = ['music', 'gamesonline', 'chat', 'ecommerce',
                'adult', 'alcohol', 'hunting', 'news', 'medical', 'religion']

# Для каждой категории (из 8000) определяем по 2 класса
# В итоге получаем 10 списков длиной 8000 ('chat', ..., 'Not_chat')
for unit in y_train:
    for clas, cat in izip(classes_list, categ_list):
        if unit == clas:
            cat.append(unit)
        else:
            cat.append('Not_' + clas)

# print len(alco_categ)
# print mus_categ, news_categ, hunt_categ, chat_categ, gam_categ

###############################################
# Обучение деревьев решений для каждой категории
# В качестве классификатора выбрано Дерево решений
clf = tree.DecisionTreeClassifier()  # min_samples_leaf=1

# Инициализация списков для хранения моделей деревьев
mus_tree, gam_tree, chat_tree, ecomrc_tree, adult_tree = clf, clf, clf, clf, clf
alco_tree, hunt_tree, news_tree, med_tree, relig_tree = clf, clf, clf, clf, clf
tree_list = [mus_tree, gam_tree, chat_tree, ecomrc_tree,
             adult_tree, alco_tree, hunt_tree, news_tree, med_tree, relig_tree]

# Обучение классификатора для каждой из категорий
for ex_tree, feat, cat in izip(tree_list, feat_list, categ_list):
    ex_tree = clf.fit(feat, cat)


#############################################
# Второй уровень: проводим вероятностную классификацию
# для 8000 тысяч сайтов из обучающей выборки

mus_pred, gam_pred, chat_pred, ecomrc_pred, adult_pred = [], [], [], [], []
alco_pred, hunt_pred, news_pred, med_pred, relig_pred = [], [], [], [], []
midpred_list = [mus_pred, gam_pred, chat_pred, ecomrc_pred,
                adult_pred, alco_pred, hunt_pred, news_pred, med_pred, relig_pred]

# Подготовленные данные подаём на вход нужным
# деревьям и сохраняем результат работы каждого из деревьев
for some_tree, train, midpred in izip(tree_list, feat_list, midpred_list):
    # Вероятностная классификация (причём внутри вер-сти [Not_cat, cat])
    predicted = some_tree.predict_proba(train)
    # print predicted
    # Заменяем содержимое списка на предсказанные значения
    midpred[0:-1] = predicted

# УЖЕ ТУТ (на втором уровне) ПЛОХИЕ ВЕРОЯТНОСТИ
# for n, m in zip (midpred_list, categ_list):
#    print n[5915], m[5915]

# Преобразование данных в таблицу 8000х10, в которой хранятся вероятности
# принадлежности каждого сайта к каждой категории
arr, X_arr = [], []
for count in range(len(mus_pred)):
    for item in midpred_list:
        arr.append(item[count][1])
    X_arr.append(arr)
    arr = []

# for tes in X_arr:
#    print tes
# print len(X_arr)


# В качестве классификатора второго уровня выбрано Дерево решений
newclf = tree.DecisionTreeClassifier()  # (min_samples_leaf=20)

# Обучение верхнеуровнего классификатора:
high_tree = newclf.fit(X_arr, y_train)

# print mus_pred[7564], gam_pred[7564], chat_pred[7564], ecomrc_pred[7564], adult_pred[7564], alco_pred[7564], hunt_pred[7564], news_pred[7564], med_pred[7564], relig_pred[7564]
# print X_mas [7564]

# print mus_pred[5555]
# print mus_pred[5555][1]
# print len(mus_pred)
##############################################


# Подготовка тестовой выборки X_test для эксперимента
# Все списки хранят свои признаки для подачи на вход деревьям
mus_test, gam_test, chat_test, ecomrc_test, adult_test = [], [], [], [], []
alco_test, hunt_test, news_test, med_test, relig_test = [], [], [], [], []
test_list = [mus_test, gam_test, chat_test, ecomrc_test,
             adult_test, alco_test, hunt_test, news_test, med_test, relig_test]

for cur in X_test:
    mus_test.append(cur[:50])
    gam_test.append(cur[50:100])
    chat_test.append(cur[100:150])
    ecomrc_test.append(cur[150:200])
    adult_test.append(cur[200:250])
    alco_test.append(cur[250:300])
    hunt_test.append(cur[300:350])
    news_test.append(cur[350:400])
    med_test.append(cur[400:450])
    relig_test.append(cur[450:])

# Эти переменные будут хранить двумерные списки вероятностей
newmus_pred, newgam_pred, newchat_pred, newecomrc_pred, newadult_pred = [], [], [], [], []
newalco_pred, newhunt_pred, newnews_pred, newmed_pred, newrelig_pred = [], [], [], [], []
newpred_list = [newmus_pred, newgam_pred, newchat_pred, newecomrc_pred, newadult_pred, newalco_pred, newhunt_pred, newnews_pred, newmed_pred, newrelig_pred]

# Подготовленные тестовые данные подаём на вход нужным
# деревьям и сохраняем результат работы каждого из деревьев
for xtree, xtest, xpred in izip(tree_list, test_list, newpred_list):
    # Вероятностная классификация (причём внутри вер-сти [Not_cat, cat])
    xpredicted = xtree.predict_proba(xtest)
    # print predicted
    # Заменяем содержимое списка на предсказанные значения
    xpred[0:-1] = xpredicted

# print newgam_pred[:3]
# print newgam_pred[-3:]


newarr, X_newarr = [], []
for count in range(len(newmus_pred)):
    for item in newpred_list:
        newarr.append(item[count][1])
    X_newarr.append(newarr)
    newarr = []

# for les in X_newarr:
#    print les
# print len(X_newarr)


# Собственно окончательная классификация на 2ом уровне
# В результате массив prob_pred хранит 2000х10 вероятностей принадлежности
# к каждой из категорий для каждого сайта
prob_pred = high_tree.predict_proba(X_newarr)

# for a in prob_pred:
#    print a
#    print max(a)
# print len(prob_pred)


# Категории с учётом неизвестной
class_labels = ['music', 'gamesonline', 'chat', 'ecommerce', 'adult',
                'alcohol', 'hunting', 'news', 'medical', 'religion', 'Unknown']

y_pred = []

# Минимальное значение вероятности для отнесения к одной из категорий
probab = 0.5
# Количество спорных/непонятных ситуаций (Unknown)
bad = 0
# Цикл для определения окончательной категории каждого из тестовых экземпляров
for cur in prob_pred:
    # Среди вероятностий принадлежности к категориям ищем максимум
    maxi = max(cur)
    # Если максимумов несколько или он меньше порогового значения,
    # итоговая категория данного экземпляра - Unknown

    if (cur.tolist().count(maxi) != 1)or(maxi < probab):
        bad += 1
        # print bad, "Problems...", maxi, is_cat.count(maxi)
        y_pred.append(class_labels[10])
    else:
        y_pred.append(class_labels[cur.tolist().index(maxi)])

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
    unknown += conf_mat[10][i]

errors = total_sites - corr_pred - unknown  # Количество ошибок
accur = corr_pred / float(total_sites)  # Общая точность

# Вывод результатов в консоль
print 'Total correct predicted:', corr_pred, ' (', accur, ')'
print 'Total Unknowns:', unknown, ' (', unknown / float(total_sites), ')'
print '\nAccuracy without unknowns:', corr_pred / float(total_sites - unknown)
print 'Errors without unknowns', (errors) / float(total_sites - unknown)

"""
################################################################

# Создание отчёта в Excel
workbook = xlsxwriter.Workbook('report (min_leaf = ' + str(
    clf.min_samples_leaf) + '; prob = ' + str(probab) + ')' + '.xlsx')  # Имя файла
worksheet = workbook.add_worksheet()
# Добавление таблицы conf_mat в отчёт
worksheet.add_table('B2:L13', {'data': conf_mat, 'autofilter': False})
# Задание имён столбцов
class_labels_true = ['true music', 'true gamesonline', 'true chat', 'true ecommerce', 'true adult',
                     'true alcohol', 'true hunting', 'true news', 'true medical', 'true religion', 'true Unknown']
worksheet.write_row('B2', class_labels_true)
# Задание имён строк
class_labels_pred = ['pred. music', 'pred. gamesonline', 'pred. chat', 'pred. ecommerce', 'pred. adult',
                     'pred. alcohol', 'pred. hunting', 'pred. news', 'pred. medical', 'pred. religion', 'pred. Unknown']
worksheet.write_column('A3', class_labels_pred)

# Задание имён строк и столбцов таблицы метрик
worksheet.write_row('A15', ['category', 'precision',
                            'recall', 'F-measure', 'allSites'])
worksheet.write_column('A16', class_labels)
# Вычисление метрик
precision, recall, fscore, support = score(y_test, y_pred)
# Запись вычисленных метрик в таблицу (на 0-ой позиции Unknown)
for i, j in zip(range(1, 11), ['B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26']):
    worksheet.write_row(j, [precision[i], recall[i], fscore[i], support[i]])
worksheet.write_row('B26', [precision[0], recall[0], fscore[0], support[0]])

# Количество ошибок; всего сайтов; верно распознанных сайтов
worksheet.write_row('A28', ['totalErrPredSites',
                            'totalSitesCount', 'totalCorrPred'])
worksheet.write_row('A29', [errors, total_sites, corr_pred])

# Средние показатели точности и полноты
worksheet.write_row('A31', ['accuracy', 'averPrecision', 'averRecall'])
worksheet.write_row('A32', [accur, sum(
    precision) / float(len(precision) - 1), sum(recall) / float(len(recall) - 1)])

# Общее количество нераспознанных сайтов и ошибок
worksheet.write_row(
    'A34', ['totalUnknowns', 'totalUnrecognized', 'totalErrWithoutUnknown'])
worksheet.write_row('A35', [unknown, total_sites - corr_pred, errors])

# Подведение итогов: показатели в виде части от общего количества сайтов
worksheet.write('A38', 'SUMMARY')
worksheet.write_row('A39', ['Accuracy', 'Errors', 'Unknowns'])
worksheet.write_row(
    'A40', [accur, errors / float(total_sites), unknown / float(total_sites)])

worksheet.write_column(
    'A42', ['Accuracy without unknowns', corr_pred / float(total_sites - unknown)])
worksheet.write_column('A45', ['Errors without unknowns',
                               errors / float(total_sites - unknown)])

workbook.close()
"""
