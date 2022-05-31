import numpy as np
import glob, os
from os import listdir
from os.path import isfile, join, isdir
from sklearn import svm
import matplotlib.pyplot as  plt
import cvxopt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from sklearn import svm
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#####################################################
authors_hapax_ratio=np.zeros((3,50))
authors_words =np.zeros((3,50))
authors_lines =np.zeros((3,50))
training_data =np.zeros ((100,3))
testing_data =np.zeros((100,3))
authors_names_training=['']*2
authors_names_testing=['']*2
flag_test=0
######################################################
def stripEnds(s):
    """ (str) -> str

    Return a new string based on s in which all letters have been
    converted to lowercase and punctuation characters have been stripped 
    from both ends. Inner punctuation is left untouched. 

    >>> stripEnds('Happy Birthday!!!')
    'happy birthday'
    >>> stripEnds("-> It's on your left-hand side.")
    ' it's on your left-hand side'
    """
    from string import punctuation
    if s.endswith("\n"or"\t"or"\r"):
        s=s[:-1]
    St=(s.strip(punctuation))
    A= St.lower()
    return A
####################################################
def hapax_legomena_ratio(text):

    Words = {}
    sentence = " ".join(text)
    sentence = stripEnds(sentence)
    splitList=sentence.split()

    for word in splitList:
        if word in Words:
            Words[word] += 1
        else:
            Words[word] = 1

    totalWords=[]
    for items in Words.items():
        totalWords.append(items)

    uniqueWords=[]
    i=0
    while i < (len(totalWords)):
        if (totalWords[i][1])==1:
            uniqueWords.append(totalWords[i])
        i=i+1

    return ( (len(uniqueWords)) / (len(splitList)) )
#####################################################
def calc_avg_words(text):
  text = text.split()
  sum = 0
  for word in text:
    sum += len(word)
  sum = sum /len(text)
  return sum
###################################################
def calc_avg_lines(text):
  text = text.split("\n")
  sum = 0
  for line in text :
    sum += len(line.split())
  sum = sum /len(text)
  return sum 
######################################################
def for_each_author (files , author_name, author_number):
  i = 0
  avg_words=[0]*50
  avg_lines=[0]*50
  for file in files : 
   if i == 50 :
     break
   #print("\nAuthor number ",author_number,"file number : ",i)
   text = open (file, 'r')
   text = text.read()
   authors_words[author_number] [i]=calc_avg_words(text)
   authors_lines[author_number][i]=calc_avg_lines(text)
   authors_hapax_ratio[author_number][i]=hapax_legomena_ratio(text)

   #print("avg words= ",authors_words[author_number][i])
   #print("avg line=",authors_lines[author_number][i])
  # print("Hapax ratio fo this file is = ", authors_hapax_ratio[author_number][i])
   i +=1
  return
##########################################################
def getAllFilesRecursive(root,t):#returns an array of  the pathes of each file
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    i =0
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d),t)
        if(t == 0): 
          authors_names_training[i]=d
        if(t == 1):
          authors_names_testing[i]=d
        if files_in_d:
            for f in files_in_d:
              files.append(f)
            for_each_author(files_in_d,d,i)

        i +=1
    return files
##______________________________________________________________________________    
##            MAIN:            
##            prepare training data :
##______________________________________________________________________________
getAllFilesRecursive("/content/drive/My Drive/Colab Notebooks/C2/training",0)
i=0
for i in range(2):
 for j in range (50): 
    training_data[i*50+j][0]=authors_words[i][j]
    training_data[i*50+j][1]=authors_lines[i][j]
    training_data[i*50+j][2]=authors_hapax_ratio[i][j]
   # print("training- author=",i,"file=",j,"is =",training_data[i*50+j])
##______________________________________________________________________________
##            prepare testing data:
##______________________________________________________________________________
getAllFilesRecursive("/content/drive/My Drive/Colab Notebooks/C2/testing",1)
i=0
for i in range(2):
  for j in range (50):
    testing_data[i*50+j][0]=authors_words[i][j]
    testing_data[i*50+j][1]=authors_lines[i][j]
    testing_data[i*50+j][2]=authors_hapax_ratio[i][j]
  #  print("testing- author=",i,"file=",j,"is =",testing_data[i*50+j])
##______________________________________________________________________________
y_test=np.zeros((50*2,1))
if(authors_names_testing[0] == authors_names_training[0]):
  for z in range(50):
    y_test[z] =0
    y_test[50+z]=1
else:
  for z in range(50):
    y_test[z] =1
    y_test[50+z]=0
##______________________________________________________________________________
y_train=np.zeros((100,1))
for i in range(2):
  for j in range(50):
    y_train[i*50+j]=i
##______________________________________________________________________________
##   
#result = svm.SVC(C=1.0, kernel='linear')
#result.fit(training_data,y_train)
#predictions_SVM = result.predict(testing_data)
#acu=accuracy_score(predictions_SVM,y_test)*100
#print("SVM Accuracy Score -> ",acu)
#--------------------------------------------------
SPACE_SAMPLING_POINTS = 100
TRAIN_POINTS = 100
X =training_data
# Define the size of the space which is interesting for the example
X_MIN = 1
X_MAX = 6  #X[:,0].max()+1
Y_MIN = 1
Y_MAX = 60 #X[:,0].max()+1
Z_MIN =0
Z_MAX = 0.1  #X[:,0].max()+1

# Generate a regular grid to sample the 3D space for various operations later
xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                         np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                         np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))
X_outliers = np.random.uniform(low=0, high=20, size=(32, 3))

# Create a OneClassSVM instance and fit it to the data
#result = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
result = svm.SVC(C=1.0, kernel='poly')
result.fit(training_data,y_train)

predictions_SVM = result.predict(testing_data)
acu=accuracy_score(predictions_SVM,y_test)*100
print("SVM Accuracy Score -> ",acu)

# Predict the class of the various input created before
y_pred_train =result.predict(training_data)
y_pred_test = result.predict(testing_data)
y_pred_outliers = result.predict(X_outliers)

# And compute classification error frequencies
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# Calculate the distance from the separating hyperplane of the SVM for the
# whole space using the grid defined in the beginning
Z = result.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Create a figure with axes for 3D plotting
fig = plt.figure()
ax = fig.gca(projection='3d')
fig.suptitle("SVM")

# Plot the different input points using 3D scatter plotting
b1 = ax.scatter(X[:50, 0], X[:50, 1], X[:50, 2], c='yellow')
b2 = ax.scatter(X[50:, 0], X[50:, 1], X[50:, 2], c='green')
#c = ax.scatter(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2], c='red')
# Plot the separating hyperplane by recreating the isosurface for the distance
# == 0 level in the distance grid computed through the decision function of the
# SVM. This is done using the marching cubes algorithm implementation from
# scikit-image.
#verts, faces = measure.marching_cubes(Z, 0)
verts, faces, norm, val = measure.marching_cubes_lewiner(Z, level=None, spacing=(1., 1., 1.),
                           gradient_direction='descent', step_size=1,
                             allow_degenerate=True, use_classic=False)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Scale and transform to actual size of the interesting volume
verts = verts * \
    [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
verts = verts + [X_MIN, Y_MIN, Z_MIN]
# and create a mesh to display
mesh = Poly3DCollection(verts[faces],
                        facecolor='orange', edgecolor='blue', alpha=0.1)
ax.add_collection3d(mesh)

# Some presentation tweaks
ax.set_xlim((0,6))
ax.set_ylim((0,60))
ax.set_zlim((0,0.1))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend([mpatches.Patch(color='orange', alpha=0.3), b1, b2, c],
          ["", "",
           "", ""],
          loc="lower left",
          prop=matplotlib.font_manager.FontProperties(size=11))

fig.show()


