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
#cursor.execute("SELECT dat_files.website_id FROM dat_files join websites on dat_files.website_id=websites.website_id WHERE ((entry_id > 100) and (entry_id < 201)) or ((entry_id > 1100) and (entry_id < 1201)) or ((entry_id > 2100) and (entry_id < 2201)) or ((entry_id > 3100) and (entry_id < 3201)) or ((entry_id > 4100) and (entry_id < 4201)) or ((entry_id > 5100) and (entry_id < 5201)) or ((entry_id > 6100) and (entry_id < 6201)) or ((entry_id > 7100) and (entry_id < 7201)) or ((entry_id > 8100) and (entry_id < 8201)) or ((entry_id > 9100) and (entry_id < 9201));")

# Only for alcohoL
cursor.execute("SELECT dat_files.website_id FROM dat_files join websites on dat_files.website_id=websites.website_id WHERE ((entry_id > 5100) and (entry_id < 5201));")

websites = cursor.fetchall()

for website_id in websites:

	#print str(website_id[0])
	command = 'python dir_img_classification.py -d ' + "/home/alant/python/source_images/" + str(website_id[0])
	print command
	os.system(command)
