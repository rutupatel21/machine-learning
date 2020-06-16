import pandas
from pandas.plotting import scatter_matrix
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Load dataset
url = "./heart.csv"

names = ["sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca", "thal" ,"target"]
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

#Describe the field thalach
print("Describe the field thalach")
print(dataset.thalach.describe())

print("frequency table")
print(dataset.thalach.value_counts())  #frequency table

print("groupby fbs")
print(dataset.groupby(by="fbs").size())
