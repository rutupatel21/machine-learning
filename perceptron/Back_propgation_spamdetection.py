from sklearn import preprocessing
import numpy as np
import pandas as pd
import nltk


##################################################
dataset = pd.read_csv('emails.csv')
dataset .columns #Index(['text', 'spam'], dtype='object')
dataset.shape  #(5728, 2)

#Checking for duplicates and removing them
dataset.drop_duplicates(inplace = True)
dataset.shape  #(5695, 2)
#Checking for any null entries in the dataset
print (pd.DataFrame(dataset.isnull().sum()))

#Using Natural Language Processing to cleaning the text to make one corpus
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Every mail starts with 'Subject :' will remove this from each text 
dataset['text']=dataset['text'].map(lambda text: text[1:])
dataset['text'] = dataset['text'].map(lambda text:re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
ps = PorterStemmer()
corpus=dataset['text'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus.values).toarray()
y = dataset.iloc[:, 1].values
##############################################





def derivative(x):
    return x * (1.0 - x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
X = []
Y = []

# read the training data
with open('Train.csv') as f:
    for line in f:
        curr = line.split(',')
        new_curr = [1]
        for item in curr[:len(curr) - 1]:
            new_curr.append(float(item))
        X.append(new_curr)
        Y.append([float(curr[-1])])
X = np.array(X)
X = preprocessing.scale(X) # feature scaling
Y = np.array(Y)

# spilt train and test data as 70/30
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# we have 3 layers: input layer, hidden layer and output layer
# input layer has 57 nodes (1 for each feature)
# hidden layer has 4 nodes
# output layer has 1 node
dim1 = len(X_train[0])
dim2 = 4
# randomly initialize the weight vectors
np.random.seed(1)
weight0 = 2 * np.random.random((dim1, dim2)) - 1
weight1 = 2 * np.random.random((dim2, 1)) - 1

for j in range(25000):
    # evaluate the output for each training email
    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))
    # calculate the error
    layer_2_error = y_train - layer_2
    # perform back propagation
    layer_2_delta = layer_2_error * derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)
    # update the weight vectors
    weight1 += layer_1.T.dot(layer_2_delta)
    weight0 += layer_0.T.dot(layer_1_delta)
    
# evaluation on the testing data
layer_0 = X_test
layer_1 = sigmoid(np.dot(layer_0,weight0))
layer_2 = sigmoid(np.dot(layer_1,weight1))
correct = 0

# if the output is > 0.5, then label as spam else no spam
for i in range(len(layer_2)):
    if(layer_2[i][0] > 0.5):
        layer_2[i][0] = 1
    else:
        layer_2[i][0] = 0
    if(layer_2[i][0] == y_test[i][0]):
        correct += 1
        
# printing the output
print ("total = ", len(layer_2))
print ("correct = ", correct)
print()

#printing confusion matrix, accuracy and classification report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = y_test 
predicted = layer_2 
results = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results)
print() 
print ('Accuracy Score :',accuracy_score(actual, predicted)) 
print()
print ('Report : ')
print (classification_report(actual, predicted))

tokenize
one pot encoding
feed to nn

Subject: photoshop , windows , office . cheap . main trending abasements darer prudently fortuitous undergone lighthearted charm orinoco taster railroad affluent pornographic cuvier irvin parkhous...	