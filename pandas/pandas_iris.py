import pandas
from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
#from matplotlib import pyplot
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#import tkinter   
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Load dataset

url = "./IRIS.csv"

names = ["sepal_length","sepal_width","petal_length","petal_width","species"]

dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)

#Print only column names in the dataset
dataset.columns.values

# head
print(dataset.head(5))
#tail
print(dataset.tail(5))


# descriptions
print("printing descriptions")
print(dataset.describe())

#Describe the field petal_width
print("Describe the field petal_width")
print(dataset.sepal_length.describe())

#frequency table
print("frequency table")
print(dataset.sepal_length.value_counts()) 

print("groupby species")
print(dataset.groupby("species").size())
