# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)


#Downloads all the images from 10000 sites and puts them in folders named by webside_id

import psycopg2
import os

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classification', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()

# Выбираем website_id и url для сайтов, изображения с которых будут скачаны
cursor.execute("SELECT dat_files.website_id, url FROM dat_files join websites on dat_files.website_id=websites.website_id WHERE (entry_id > 7000) and (entry_id < 10001);")

websites = cursor.fetchall()


# all_features = []  # хранит таблицу 10000 строк по 500 признаков
for site in websites:
    folder = "/home/alant/python/images/" + str(site[0])
    print site[1]
    print folder
    command = 'python download_img.py ' + site[1] + ' ' + folder
    os.system("mkdir /home/alant/python/images/" + str(site[0]))
    os.system(command)
# print all_features

# print (websites[1][1])


# Затык произошёл после "165978 | http://121musicblog.com". Возможно виновен 165981 | http://123musicstars.com
# Поэтому скрипт прогоняем с entry_id = 28 до 1001

# Попался сайт по продаже ковров - и там много картинок ковров 2405229 (на 73ой я прервал загрузку с него)