#!/usr/bin/env python
# coding: utf-8

# # OS Module in Python

# <p>**os** module of python is prodigiously used for loading the data on the Memory and preprocessing tasks in Machine Learning. Preprocessing may differ depending on what one intends to do. One should be familiar with the ways in which **os** module can be used to suit their needs. </p>
# 
# <h3>To list a few functions that are used:</h3>
# - *os.listdir*
# - *os.walk*
# - *os.mkdir*
# 
# <h3>Few attributes used:</h3>
# - *os.path.sep*

# In[12]:


import os


# In[13]:


folder_path = '.' # Single dot indicates the path of the CURRENT folder/directory. In UNIX term, single dot is a hard-link to the current directory.


# <h3>Print the list of names of all the files and folders/directories present in the CURRENT folder/directory only</h3>
# 
# 

# In[14]:


print(os.listdir('.'))


# <h3>Observe the output of os.walk</h3>

# In[20]:


labels = []
validExtensions = ['jpg', 'jpeg', 'png', 'bmp']
for pathName, folderNames, fileNames in os.walk('./datasets/animals'):
    print(pathName, folderNames, fileNames)
    print()


# <h3>Print all the path names, folder and file names present in datasets folder (recursively)</h3>
# 

# In[21]:


print(os.listdir('./datasets'))
for pathName, folderNames, fileNames in os.walk('./datasets'):
    print(pathName, folderNames, fileNames)


# <h3>Print the path name along with file name for all the files present in the datasets/animals folder (recursively)</h3>
# 

# In[22]:


imagePaths = []
for pathName, folderNames, fileNames in os.walk('./datasets/animals'): #http://www.bogotobogo.com/python/python_traversing_directory_tree_recursively_os_walk.php
    for fileName in fileNames:
        imagePaths.append(pathName+'/'+fileName)
        
print(imagePaths)


# <h3>Keep only those paths with fileNames ending with .jpg, .jpeg, .png, .bmp</h3>

# In[23]:


imagePaths = []
validExtensions = ['jpg', 'jpeg', 'png', 'bmp']
for pathName, folderNames, fileNames in os.walk('./datasets/animals'):
    for fileName in fileNames:
        if fileName.split(".")[-1] in validExtensions: #https://pythonprogramminglanguage.com/split-string/
            imagePaths.append(pathName+'/'+fileName)
            
print(imagePaths)


# <h3>Extract the class label assuming that our path has the following format :</h3>
# <h4>/path/to/dataset/{class}/{image}.jpg<h4>
# <h3>Hint: Look at the output of imagePaths above</h3>

# In[24]:


labels = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    if  label not in labels: #os.path.sep refers to path separator. 
        labels.append(label) #On Windows, path separator is '\', where as on Ubuntu it is '/'

print(labels)


# In[ ]:





# In[ ]:




