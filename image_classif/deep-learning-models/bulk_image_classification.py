# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)

# import the necessary packages
import psycopg2
import os

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classification', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()

# Выбираем website_id для сайтов, изображения с которых будут классифицированы
cursor.execute("SELECT dat_files.website_id FROM dat_files join websites on dat_files.website_id=websites.website_id WHERE ((entry_id > 0) and (entry_id < 11)) or ((entry_id > 1000) and (entry_id < 1011)) or ((entry_id > 2000) and (entry_id < 2011)) or ((entry_id > 3000) and (entry_id < 3011)) or ((entry_id > 4000) and (entry_id < 4011)) or ((entry_id > 5000) and (entry_id < 5011)) or ((entry_id > 6000) and (entry_id < 6011)) or ((entry_id > 7000) and (entry_id < 7011)) or ((entry_id > 8000) and (entry_id < 8011)) or ((entry_id > 9000) and (entry_id < 9011));")

websites = cursor.fetchall()

for website_id in websites:

	#print str(website_id[0])
	command = 'python dir_img_classification.py -d ' + "/home/alant/python/source_images/" + str(website_id[0])
	print command
	os.system(command)
