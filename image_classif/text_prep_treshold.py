# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)

# import the necessary packages
import psycopg2
import os
import glob
import re
from collections import Counter
import math
import operator

from itertools import *
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import xlsxwriter
from sklearn.metrics import precision_recall_fscore_support as score

import pydotplus
import datetime

# Граница вероятности для отсесения класса к итоговому списку классов сайта
TRESHOLD_PROB = 0.31
# Минимальное колиество сайтов в категории, в которых должен содержаться класс, чтобы считаться весомым при подсчёте IDF
TRESHOLD_IDF = 2
# Количество лучших терминов из каждой категории для формирования итогового словаря (размером FOR_BASIS*10)
FOR_BASIS = 13
# Минимальное количество листьев дерева решений
LEAVES_COUNT = 60
# Минимальное значение вероятности для отнесения к одной из категорий на втором этапе классификации (после деревьев)
MIN_PROBAB = 0.3

def file_to_list_with_treshold (full_path_to_file, treshold):
	""" Функция преобразует файл, заданный в виде строки (полный путь к файлу)
	в список классов изображений, вероятность которых выше заданного порога (treshold) """

	# Переменная для очистки классов и вероятностей от лишних символов 
	regex = re.compile('^ ?\[?u|[^a-zA-Z0-9_.]')

	# Переменная для хранения результирующего списка значимых классов
	result_list = []

	# Открытие файла и построчное считывание содержимого в список "lines"
	with open(full_path_to_file) as f: lines = [line.rstrip('\n') for line in f]

	# Для каждой строки
	for cur_line in lines:
		
		# Разбиваем строку на элементы и помещаем в список
		line_list = cur_line.split(",")
		# print line_list
		
		# Для каждого второго элемента строки, т.е вероятности
		for i, elem in enumerate(line_list):
			if (i % 2 != 0):
				# Очищаем от лишних символов
				elem = regex.sub('', elem)
				# Проверяем, превышает ли вероятность класса заданный treshold
				if float(elem)>treshold:
					#print regex.sub('', line_list[i-1])
					# Если превышает, помещаем в результирующий список класс, соответствующий вер-сти
					result_list.append(regex.sub('', line_list[i-1]))

	#print result_list
	return result_list


#file = "/home/alant/python/nirs/image_classif/results/VGG16_top-5_banner_4/14995.txt"
#list_14995 = file_to_list_with_treshold (file, 0.2)
#print list_14995

######################################
# Функция составления хэш-таблицы из списка слов: {'some_class' : 1, 'another_class' : 1, ...}
def compute_tf(text):
    tf_text = {i: 1 for i in text}
    return tf_text
######################################

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classification', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()



# Функция для составления хэш-таблицы со значениями tf классов для всей категории
# Возвращает (hash-table, not_empty_count)
def tf_for_category(entry_id_from,entry_id_to):

	# Расчёт словаря TF для категории music
	cursor.execute("SELECT website_id from dat_files where ((entry_id > " + str(entry_id_from) + ") and (entry_id < " + str(entry_id_to) + "));")

	# Переменная для хранения итоговой хэш-таблицы с кол-вом классов, встретившихся в категории
	result_dict = Counter({})
	# Количество не пустых списков после подготовки текста
	not_empty_count = 0
	# Общий для категории список для хранения результирующих списков классов каждой веб-страницы
	site_raw_features = []
	website_ids = cursor.fetchall()
	for website_id in website_ids:
		#print website_id[0]
		path_to_file = '/home/alant/python/nirs/image_classif/results/VGG16_top-5_banner_4/' + str(website_id[0]) + ".txt"
		#print path_to_file
		site_text = file_to_list_with_treshold (path_to_file, TRESHOLD_PROB)
		# Подсчёт количества непустых сайтов, имеющих значимые классы картинок
		if site_text:
			not_empty_count += 1
			site_raw_features.append(site_text)

		site_dict = Counter(compute_tf(site_text))
		result_dict += site_dict

	return result_dict, not_empty_count, site_raw_features
	#print result_dict
	#print len(website_ids)
	#print not_empty_count


music_tfs = tf_for_category(0,1001)
#print music_tfs[2]
#print len(music_tfs[2])

gamesonline_tfs = tf_for_category(1000,2001)
#print gamesonline_tfs

chat_tfs = tf_for_category(2000,3001)
#print chat_tfs

ecommerce_tfs = tf_for_category(3000,4001)
#print ecommerce_tfs

adult_tfs = tf_for_category(4000,5001)
#print adult_tfs

alcohol_tfs = tf_for_category(5000,6001)
#print alcohol_tfs

hunting_tfs = tf_for_category(6000,7001)
#print hunting_tfs

news_tfs = tf_for_category(7000,8001)
#print news_tfs

medical_tfs = tf_for_category(8000,9001)
#print medical_tfs

religion_tfs = tf_for_category(9000,10001)
#print religion_tfs


# Общее количество непустых результирующих списков из 10К сайтов
#total_count = music_tfs[1]+gamesonline_tfs[1]+chat_tfs[1]+ecommerce_tfs[1]+adult_tfs[1]+alcohol_tfs[1]+hunting_tfs[1]+news_tfs[1]+medical_tfs[1]+religion_tfs[1]
#print total_count

# Определяем количество сайтов из каждой категории для дальнейшей обработки
FROM_EACH_SITE = min(music_tfs[1], gamesonline_tfs[1], chat_tfs[1], ecommerce_tfs[1], adult_tfs[1], alcohol_tfs[1], hunting_tfs[1], news_tfs[1], medical_tfs[1], religion_tfs[1])
print music_tfs[1], gamesonline_tfs[1], chat_tfs[1], ecommerce_tfs[1], adult_tfs[1], alcohol_tfs[1], hunting_tfs[1], news_tfs[1], medical_tfs[1], religion_tfs[1]
print min(music_tfs[1], gamesonline_tfs[1], chat_tfs[1], ecommerce_tfs[1], adult_tfs[1], alcohol_tfs[1], hunting_tfs[1], news_tfs[1], medical_tfs[1], religion_tfs[1])

# Словарь, содержаший все классы и кол-во сайтов, на которых они встретились (с заданной вер-стью)
#total_dict = music_tfs[0]+gamesonline_tfs[0]+chat_tfs[0]+ecommerce_tfs[0]+adult_tfs[0]+alcohol_tfs[0]+hunting_tfs[0]+news_tfs[0]+medical_tfs[0]+religion_tfs[0]
#print total_dict


all_tfs = [music_tfs, gamesonline_tfs, chat_tfs, ecommerce_tfs, adult_tfs, alcohol_tfs, hunting_tfs, news_tfs, medical_tfs, religion_tfs]


def idf_for_category(category_tfs, treshold_idf):

	IDF_dict = Counter({})

	# Подсчёт IDF для всех классов категории и запись в единый словарь
	for term in category_tfs[0]:
		#print i, total_dict[i], math.log10(total_count/total_dict[i])
		#print term, category_tfs[0][term]
		num_of_categs = 0
		# Если класс встречается в категории больше treshold_idf раз, считаем его IDF
		if (category_tfs[0][term] > treshold_idf-1):
			# Проверяем наличие класса более treshold_idf раз во всех категориях
			for cur_categ in all_tfs:
				#print cur_categ[0][term]
				if (cur_categ[0][term] > treshold_idf-1):
					num_of_categs += 1

			#if (num_of_categs > 0):
			term_idf = math.log(len(all_tfs)/float(num_of_categs))
		
		# Иначе его IDF для категории=0
		else: term_idf = 0
		
		#print 'IDF', term_idf
		IDF_dict[term] = term_idf

	return IDF_dict



#print idf_for_category(music_tfs,5).most_common(20)




def produce_tf_idf_for_category(category_tfs):

	tf_idf_dict = Counter({})

	IDF_dict = idf_for_category(category_tfs,TRESHOLD_IDF)

	for term in category_tfs[0]:
		
		tf = float(category_tfs[0][term])/category_tfs[1]
		#print "Term, TF, IDF: ", term, tf, IDF_dict[term]
		#print term, tf, IDF_dict[term]

		tf_idf = tf * IDF_dict[term]
		#print "TF-IDF:			", tf_idf, '\n'
		#print tf_idf
		tf_idf_dict[term] = tf_idf

	return tf_idf_dict



# n - сколько слов брать из каждой категории в итоговый словарь
def produce_basis_list(list_of_tfs, n):

	# Переменная для хранения словаря (n-лучших слов каждой категории)
	basis_list = []

	for tfs in list_of_tfs:

		cur_tf_idf = produce_tf_idf_for_category(tfs)

		#print '\n'
		i=0
		for elem in cur_tf_idf.most_common(n):
			#print elem[0], elem[1]
			if elem[0] not in basis_list:
				basis_list.append(elem[0])
			else:
				i+=1
				#print elem[0]

		#print 'Already in list: ', i
		#if i>0: print cur_tf_idf.most_common(10+i)
		k=1
		while (k <= i):
			next_elem=cur_tf_idf.most_common(n+k)[-1:][0][0]

			if next_elem not in basis_list:
				basis_list.append(next_elem)
				#print next_elem
			else: 
				#print 'oh sheet', next_elem
				i+=1
			k+=1

		#print basis_list

	return basis_list


basis = produce_basis_list(all_tfs, FOR_BASIS)
print basis
#print len(basis)

# Список для хранения свойств размером 5000x100 (по 500 сайтов из категории, при n=10)
all_feat_list = []

# Формирование двумерного списка всех свойств (в бинарном виде)
for cur_tf in all_tfs:
	category_binary_list=[]
	
	cur_sorted=sorted(cur_tf[2], key=len)
	#print '\n'
	for site_list in cur_sorted:
		#print site_list
		site_binary_list=[]
		for elem in basis:
			if elem in site_list: site_binary_list.append(1)
			else: site_binary_list.append(0)

		category_binary_list.append(site_binary_list)
# самые длинные 500
	category_binary_list=category_binary_list[-FROM_EACH_SITE:]

		#print site_binary_list
	#all_feat_list.append(site_binary_list)
	for elem_list in category_binary_list:
		#print elem_list
		all_feat_list.append(elem_list)

#print all_feat_list
#print len(all_feat_list)

#print religion_tfs[2][412]
#print len(religion_tfs[2][412])
#print sorted(religion_tfs[2], key=len)
#print religion_tfs[2].sort(key=len)


#print religion_tfs[2][498]

categ_vals = []
categories_list = ['music', 'gamesonline', 'chat', 'ecommerce',
                'adult', 'alcohol', 'hunting', 'news', 'medical', 'religion']

for category in categories_list:
	for i in xrange(FROM_EACH_SITE):
		categ_vals.append(category)

#print categ_vals[4000],categ_vals[4499],categ_vals[4500],categ_vals[4999],categ_vals[3000]
#print len(categ_vals)










#####################################################################

X_train, X_test, y_train, y_test = train_test_split(
    all_feat_list, categ_vals, test_size=0.2, stratify=categ_vals)

#print X_train
#print len(X_train)
#print len(y_train)

# Формируем таблицы 4000х10 признаков для каждой категории
# То есть для категории есть только её признаки у каждого из 4К сайтов
mus_feat, gam_feat, chat_feat, ecomrc_feat, adult_feat = [], [], [], [], []
alco_feat, hunt_feat, news_feat, med_feat, relig_feat = [], [], [], [], []
feat_list = [mus_feat, gam_feat, chat_feat, ecomrc_feat,
             adult_feat, alco_feat, hunt_feat, news_feat, med_feat, relig_feat]

for cur in X_train:
    mus_feat.append(cur[:FOR_BASIS])
    gam_feat.append(cur[FOR_BASIS:2*FOR_BASIS])
    chat_feat.append(cur[2*FOR_BASIS:3*FOR_BASIS])
    ecomrc_feat.append(cur[3*FOR_BASIS:4*FOR_BASIS])
    adult_feat.append(cur[4*FOR_BASIS:5*FOR_BASIS])
    alco_feat.append(cur[5*FOR_BASIS:6*FOR_BASIS])
    hunt_feat.append(cur[6*FOR_BASIS:7*FOR_BASIS])
    news_feat.append(cur[7*FOR_BASIS:8*FOR_BASIS])
    med_feat.append(cur[8*FOR_BASIS:9*FOR_BASIS])
    relig_feat.append(cur[9*FOR_BASIS:])
#print ecomrc_feat

# Инициализация списков для хранения категорий
mus_categ, gam_categ, chat_categ, ecomrc_categ, adult_categ = [], [], [], [], []
alco_categ, hunt_categ, news_categ, med_categ, relig_categ = [], [], [], [], []
categ_list = [mus_categ, gam_categ, chat_categ, ecomrc_categ, adult_categ,
              alco_categ, hunt_categ, news_categ, med_categ, relig_categ]

classes_list = ['music', 'gamesonline', 'chat', 'ecommerce',
                'adult', 'alcohol', 'hunting', 'news', 'medical', 'religion']

# Для каждой категории (из 4000) определяем по 2 класса
# В итоге получаем 10 списков длиной 4000 ('chat', ..., 'Not_chat')
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
clf = tree.DecisionTreeClassifier(min_samples_leaf=LEAVES_COUNT)

# Инициализация списков для хранения моделей деревьев
mus_tree, gam_tree, chat_tree, ecomrc_tree, adult_tree = clf, clf, clf, clf, clf
alco_tree, hunt_tree, news_tree, med_tree, relig_tree = clf, clf, clf, clf, clf
tree_list = [mus_tree, gam_tree, chat_tree, ecomrc_tree,
             adult_tree, alco_tree, hunt_tree, news_tree, med_tree, relig_tree]

# Обучение классификатора для каждой из категорий
for cur_tree, feat, cat in izip(tree_list, feat_list, categ_list):
    cur_tree = clf.fit(feat, cat)


# Графическое представление построенного дерева
dot_data = tree.export_graphviz(tree_list[5], out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(
    "tree_" + datetime.datetime.now().strftime("%H:%M:%S") + ".pdf")




# Подготовка тестовой выборки X_test для эксперимента
# Все списки хранят свои признаки для подачи на вход деревьям
mus_test, gam_test, chat_test, ecomrc_test, adult_test = [], [], [], [], []
alco_test, hunt_test, news_test, med_test, relig_test = [], [], [], [], []
test_list = [mus_test, gam_test, chat_test, ecomrc_test,
             adult_test, alco_test, hunt_test, news_test, med_test, relig_test]

for cur in X_test:
    mus_test.append(cur[:1*FOR_BASIS])
    gam_test.append(cur[1*FOR_BASIS:2*FOR_BASIS])
    chat_test.append(cur[2*FOR_BASIS:3*FOR_BASIS])
    ecomrc_test.append(cur[3*FOR_BASIS:4*FOR_BASIS])
    adult_test.append(cur[4*FOR_BASIS:5*FOR_BASIS])
    alco_test.append(cur[5*FOR_BASIS:6*FOR_BASIS])
    hunt_test.append(cur[6*FOR_BASIS:7*FOR_BASIS])
    news_test.append(cur[7*FOR_BASIS:8*FOR_BASIS])
    med_test.append(cur[8*FOR_BASIS:9*FOR_BASIS])
    relig_test.append(cur[9*FOR_BASIS:])

# Эти переменные будут хранить двумерные списки вероятностей
mus_pred, gam_pred, chat_pred, ecomrc_pred, adult_pred = [], [], [], [], []
alco_pred, hunt_pred, news_pred, med_pred, relig_pred = [], [], [], [], []
pred_list = [mus_pred, gam_pred, chat_pred, ecomrc_pred,
             adult_pred, alco_pred, hunt_pred, news_pred, med_pred, relig_pred]


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
class_labels = ['music', 'gamesonline', 'chat', 'ecommerce', 'adult',
                'alcohol', 'hunting', 'news', 'medical', 'religion', 'Unknown']
y_pred = []

# Минимальное значение вероятности для отнесения к одной из категорий
probab = MIN_PROBAB
# Количество спорных/непонятных ситуаций (Unknown)
bad = 0
# Цикл для определения окончательной категории каждого из тестовых экземпляров
for mus, gam, ch, ec, ad, al, hn, nw, med, rel in izip(mus_pred, gam_pred, chat_pred, ecomrc_pred, adult_pred, alco_pred, hunt_pred, news_pred, med_pred, relig_pred):
    # Среди вероятностий принадлежности к категориям ищем максимум
    is_cat = [mus[1], gam[1], ch[1], ec[1], ad[
        1], al[1], hn[1], nw[1], med[1], rel[1]]
    maxi = max(is_cat)
    # Если максимумов несколько или он меньше порогового значения,
    # итоговая категория данного экземпляра - Unknown
    if (is_cat.count(maxi) != 1) or (maxi < probab):
        bad += 1
        # print bad, "Problems...", maxi, is_cat.count(maxi)
        y_pred.append(class_labels[10])
    else:
        y_pred.append(class_labels[is_cat.index(maxi)])

# print y_pred


# Формирование матрицы и вывод в консоль
orig_conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
report = classification_report(y_test, y_pred, labels=class_labels)
print "\nConfusion matrix (with probability):\n\n", orig_conf_mat
print "\n", report

f1_score = f1_score(y_test, y_pred, average='weighted', labels=class_labels[:10])
#print 'f1-score:', f1

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
class_labels = sorted(class_labels, key=lambda s: s.lower())
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