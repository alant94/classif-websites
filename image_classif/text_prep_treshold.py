# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)

# import the necessary packages
import psycopg2
import os
import glob
import re
from collections import Counter

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


# Функция для составления хэш-таблицы со значениями tf для всей категории
# Возвращает (hash-table, not_empty_count)
def tf_for_category(entry_id_from,entry_id_to):

	# Расчёт словаря TF для категории music
	cursor.execute("SELECT website_id from dat_files where ((entry_id > " + str(entry_id_from) + ") and (entry_id < " + str(entry_id_to) + "));")

	# Переменная для хранения итоговой хэш-таблицы с кол-вом классов, встретившихся в категории
	result_dict = Counter({})
	# Количество не пустых списков после подготовки текста
	not_empty_count = 0

	website_ids = cursor.fetchall()
	for website_id in website_ids:
		#print website_id[0]
		path_to_file = '/home/alant/python/nirs/image_classif/results/VGG16_top-5_banner_4/' + str(website_id[0]) + ".txt"
		#print path_to_file
		site_text = file_to_list_with_treshold (path_to_file, 0.2)
		# Подсчёт количества сайтов без классов
		if site_text:
			not_empty_count = not_empty_count + 1
		site_dict = Counter(compute_tf(site_text))
		result_dict = result_dict + site_dict

	return result_dict, not_empty_count
	#print result_dict
	#print len(website_ids)
	#print not_empty_count


music_tfs = tf_for_category(0,1001)
#print music_results

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
print religion_tfs



# print list_of_files
"""
# Переменная для хранения итоговой хэш-таблицы с кол-вом классов, встретившихся в категории
result_dict = Counter({})

for cur_file in list_of_files:
	site_text = file_to_list_with_treshold (cur_file, 0.2)
	site_dict = Counter(compute_tf(site_text))
	result_dict = result_dict + site_dict

print len(list_of_files)
print result_dict
# print list_of_files
"""

#####################################################################

