# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)

# import the necessary packages
import psycopg2
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Соединение с базой данных
connect = psycopg2.connect(
    database='Classification', user='postgres', host='localhost', password='postgres')
cursor = connect.cursor()


#listOfImg = os.listdir("/home/alant/python/nirs/image_classif/results/")
listOfImg = glob.glob('/home/alant/python/nirs/image_classif/results/[0-9]*.txt')
#print listOfImg

# Запрос для вывода категории сайта по его website_id
# SELECT category_name FROM websites join categories on websites.category_id=categories.category_id WHERE website_id=27485;

# Объявление словарей для хранения статистики:
# Для каждой категории получим классы изображений и количество таких изображений
# в виде {'класс1':2, 'класс2':4, ... , 'класс9':1}
mus_data, gam_data, chat_data, ecomrc_data, adult_data = {}, {}, {}, {}, {}
alco_data, hunt_data, news_data, med_data, relig_data = {}, {}, {}, {}, {}

# Словарь для сопоставления категории сайта и соответствующего массива данных 
class_to_list = {
    'music': mus_data,
    'gamesonline': gam_data,
    'chat': chat_data,
    'ecommerce': ecomrc_data,
    'adult': adult_data,
    'alcohol': alco_data,
    'hunting': hunt_data,
    'news': news_data,
    'medical': med_data,
    'religion': relig_data
}

# TEST VARS
mus=0
gam=0
ch=0
al=0
for file in listOfImg:
	website_id = (file.split("/")[-1]).split(".")[0]

	cursor.execute("SELECT category_name FROM websites join categories on websites.category_id=categories.category_id WHERE website_id=" + website_id + ";")
	category_name = cursor.fetchall()
	
	#print category_name[0][0]
	#print (category_name[0][0]=="music")
	'''
	if (category_name[0][0]=="music"):
		mus+=1
		print category_name[0][0]
		print mus
	elif (category_name[0][0]=="gamesonline"):
		gam+=1
		print category_name[0][0]
		print gam
	elif (category_name[0][0]=="chat"):
		ch+=1
		print category_name[0][0]
		print ch
		print website_id
	elif (category_name[0][0]=="alcohol"):
		al+=1
		print category_name[0][0]
		print al
		#print website_id
	
	else: print 'WTF!!!!'
	'''	


	with open(file) as f: lines = [line.rstrip('\n') for line in f]
	
	for img_class in lines:
		if not img_class in class_to_list[category_name[0][0]]:
			class_to_list[category_name[0][0]][img_class] = 1
		else:
			class_to_list[category_name[0][0]][img_class] += 1

	#if (category_name[0][0]=="hunting"):
		#print website_id, category_name[0][0]
		#print class_to_list[category_name[0][0]]

	


df = pd.DataFrame(data=mus_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/music.xlsx')
#df2 = pd.DataFrame(data=gam_data, index=[2])
#df.loc[len(df)]=gam_data
#df.to_csv('/tmp/music.csv')
#df.to_excel('/tmp/music.xlsx')

df = pd.DataFrame(data=gam_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/gamesonline.xlsx')

df = pd.DataFrame(data=chat_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/chat.xlsx')

df = pd.DataFrame(data=ecomrc_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/ecommerce.xlsx')

df = pd.DataFrame(data=adult_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/adult.xlsx')

df = pd.DataFrame(data=alco_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/alco_data.xlsx')

df = pd.DataFrame(data=hunt_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/hunting.xlsx')

df = pd.DataFrame(data=news_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/news.xlsx')

df = pd.DataFrame(data=med_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/medical.xlsx')

df = pd.DataFrame(data=relig_data, index=[1])  # transpose to look just like the sheet above
df.to_excel('/tmp/religion.xlsx')


#D = mus_data
#plt.bar(range(len(D)), D.values(), align='center')
#plt.xticks(range(len(D)), D.keys())
#plt.show()


'''
x = {}

var = ['key3','key1','key1','key3','key4']

for i in var:
    if not i in x:
        x[i] = 1
    else:
        x[i] += 1

print x
'''