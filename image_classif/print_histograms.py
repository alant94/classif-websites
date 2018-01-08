# This Python file uses the following encoding: utf-8
# Author: Aliaksei Antonau (alant4741@mail.ru)

# import the necessary packages

import numpy as np
import matplotlib.pyplot as plt
#import plotly.plotly as py

fileName = "/home/alant/python/nirs/image_classif/dict_results.txt"
file = open(fileName, 'r')

dict_results = np.loadtxt(file)

prob = []
idf = []
basis = []



for elem in dict_results:
	if (elem[3]>0.8) and (elem[4]>0.2):
		prob.append(elem[0])
		idf.append(elem[1])
		basis.append(elem[2])


print prob, '\n'
print idf, '\n'
print basis, '\n'

#print len(prob)
#print len(idf)
#print len(basis)


fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

axs[0].hist(prob)
axs[1].hist(idf)
axs[2].hist(basis)

plt.show()

'''
plt.hist(prob)
plt.title("Probability Histogram")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()

#plt.hist(prob, bins='auto')
#plt.title("Probability Histogram")
#plt.xlabel("Probability")
#plt.ylabel("Frequency")
#plt.show()

plt.hist(idf)
plt.title("IDF Histogram")
plt.xlabel("IDF")
plt.ylabel("Frequency")
plt.show()

plt.hist(basis)
plt.title("BASIS Histogram")
plt.xlabel("BASIS")
plt.ylabel("Frequency")
plt.show()
'''