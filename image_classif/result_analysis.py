# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)

# import the necessary packages
import psycopg2
import os
import glob

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classification', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()


#listOfImg = os.listdir("/home/alant/python/nirs/image_classif/results/")
listOfImg = glob.glob('/home/alant/python/nirs/image_classif/results/[0-9]*.txt')
print listOfImg

# Запрос для вывода категории сайта по его website_id
# SELECT category_name FROM websites join categories on websites.category_id=categories.category_id WHERE website_id=27485;