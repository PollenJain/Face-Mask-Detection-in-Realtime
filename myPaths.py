#!/usr/bin/env python

import os

def get_image_path_list(datasetPath):
	imagePaths = []
	validExtensions = ['jpg', 'jpeg', 'png', 'bmp']
	for pathName, folderNames, fileNames in os.walk(datasetPath):
		for fileName in fileNames:
			if fileName.split(".")[-1] in validExtensions: #https://pythonprogramminglanguage.com/split-string/
				imagePaths.append(pathName+'/'+fileName)
				
	#print(imagePaths)

	return imagePaths




# In[ ]:





# In[ ]:




