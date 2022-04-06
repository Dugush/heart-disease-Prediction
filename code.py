# Inputs and Visualizations
## Univariate Plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inlin

fileName = 'heart_disease_health_indicators.csv'
namesCol = [
'highBp',
'highChol',
'cholCheck',
'bmi',
'smoker',
'stroke',
'diabetes',
'physActivity',
'fruits',
'veggies',
'hvyAlcoholConsump',
'anyHealthcare',
'noDocbcCost',
'genHlth',
'mentHlth',
'physHlth',
'diffWalk',
'sex',
'age',
'education',
'income',
'class'
]
data = pd.read_csv(fileName, names=namesCol)
data.head()
## Histogram
plt.rc('font', size = 40)
data.hist(figsize=(70,70))
plt.savefig("pimaHist.jpg")
plt.show()

## Density Plot
from matplotlib.pyplot import figure
plt.rcParams["figure.figsize"] = (15,15)
# Reset to default size:
# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rc('font', size = 10)
data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
plt.show()

## Skew Values
skew = data.skew()
print(skew)

## Correlations
filename = 'heart_disease_health_indicators.csv'
names = ['highBp', 'highChol', 'cholCheck', 
'bmi', 'smoker', 'stroke', 'diabetes','physActivity', 
'fruits', 'veggies','hvyAlcoholConsump','anyHealthcare', 'noDocbcCost',
'genHlth', 'mentHlth', 'physHlth', 'diffWalk', 'sex',
'age', 'education', 'income','class']
data = pd.read_csv(filename, names=names)
correlations = data.corr()
print(correlations)
print("Correlation Matrix")
fig = plt.figure(figsize=(10,10))
3
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,22,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
print("Correlation Table")
# Correlation Table, note this does not export easily
corr = data.corr()
corr.style.background_gradient().set_precision(2)
## Class Breakdown
class_counts = data.groupby('class').size()
print(class_counts)
# Note, get to understand your outputs
print("-------------------------")
print("0 : No illness")
print("1 : Illness")

## Data Types
import matplotlib.pyplot as plt
import pandas as pd
# USeful for matplotlib in JN's
%matplotlib inline
filename = 'heart_disease_health_indicators.csv'
names = ['highBp', 'highChol', 'cholCheck', 
'bmi', 'smoker', 'stroke', 'diabetes','physActivity', 
'fruits', 'veggies','hvyAlcoholConsump','anyHealthcare', 'noDocbcCost',
'genHlth', 'mentHlth', 'physHlth', 'diffWalk', 'sex',
'age', 'education', 'income','class']
data = pd.read_csv(filename, names=names)
types=data.dtypes
print(types)

## Data shape
print(data.shape)

## Check for missing data
data.isna().sum()

## Drop Duplicated rows
duplicated_rows = data[data.duplicated()]
print(f'we have {duplicated_rows.shape[0]} duplicated rows in our data.')

data.loc[data.duplicated(), :]

## Drop duplicated rows
data.drop_duplicates(inplace=True)
print(f'data shape after drop duplicated rows : {data.shape}')

## Split X and y
X = data.drop(columns='class')
y = pd.DataFrame(data['class'])

print(f'X shape : {X.shape}')
print(f'y shape : {y.shape}')

## Target value distribution
y.value_counts()
 ## Imbalanced dataset
  # distribution of data in each class
import seaborn as sns
sns.countplot(x="class", data=y)
plt.title("Distribution of data in each class")
plt.show()

## Split train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify= y, random_state=42)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

## Standardization
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
X_train[X_train.columns] = rs.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = rs.transform(X_test[X_test.columns])

# Attribute_Selection
## Using Pearson Correlation

import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(16,14))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

#### with the following function we can select highly correlated features
### it will remove the first feature that is correlated with anything other feature
# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
  
  corr_features = correlation(X_train, 0.2)
len(set(corr_features))

corr_features

X_train.drop(corr_features,axis=1)
X_test.drop(corr_features,axis=1)

## Recursive Feature Elimination
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=45957, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# fit RFE
rfe.fit(X, y)
# summarize all features
for i in range(X.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
    
    # Model Performance
    import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# SK learn Models-> https://scikit-learn.org/stable/
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# sklearn hold out
from sklearn.model_selection import train_test_split

filename = 'heart_disease_health_indicators.csv'
colNames = ['highBp','cholCheck','bmi','smoker','stroke','physActivity','fruits','hvyAlcoholConsump','anyHealthcare','sex']


X = array[:,0:9]
Y = array[:,9]

# Set the siz of the training and test set (in percentage)
test_size = 0.33

# random seed for the data
seed = 1

# Returns 4 lists
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)

print("Naive Bayes:\n------------------------------------")
model = GaussianNB()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy:", round(result*100.0,2), "%")

print("\n\n\nLogistic Regression:\n------------------------------------")
model2 = LogisticRegression()
model2.fit(X_train, Y_train)
result2 = model2.score(X_test, Y_test)

print("Accuracy:", round(result2*100.0,2), "%")
data = pd.read_csv(filename, names=colNames)
array = data.values

## Hold-out method
import pandas as pd

# sklearn 10FCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# SK learn Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

filename = 'heart_disease_health_indicators.csv'
colNames = ['highBp','cholCheck','bmi','smoker','stroke','physActivity','fruits','hvyAlcoholConsump','anyHealthcare','sex']

data = pd.read_csv(filename, names=colNames)
array = data.values

X = array[:,0:9]
Y = array[:,9]

# Folds and seed
num_folds = 10
seed = None

print("Naive Bayes:\n------------------------------------")
kfold = KFold(n_splits=num_folds, random_state=seed)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:", round(results.mean()*100.0,2), "\tStandard Deviation", round(results.std()*100.0,2))

print("\n\n\nLogistic Regression:\n------------------------------------")
kfold = KFold(n_splits=num_folds, random_state=seed)

model2 = LogisticRegression()

results2 = cross_val_score(model2, X, Y, cv=kfold)
print("Accuracy:", round(results2.mean()*100.0,2),"\tStandard Deviation",
round(results2.std()*100.0,2))

# Model Performance - Classification
## Confusion Matrix (Method 1 - 10fold Cross Validation)
import pandas as pd
# sklearn 10FCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# confusion matrix
from sklearn.metrics import confusion_matrix

# SK learn Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

filename = 'heart_disease_health_indicators.csv'
colNames = ['highBp','cholCheck','bmi','smoker','stroke','physActivity','fruits','hvyAlcoholConsump','anyHealthcare','sex']

data = pd.read_csv(filename, names=colNames)

array = data.values
X = array[:,0:9]
Y = array[:,9]
# Folds and seed
num_folds = 10

seed = None
print("Naive Bayes:\n------------------------------------")
kfold = KFold(n_splits=num_folds, random_state=seed)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:", round(results.mean()*100.0,2)," Standard Deviation",round(results.std()*100.0,2))

# over all confusion matrix
y_pred = cross_val_predict(model, X, Y, cv=10)
conf_mat = confusion_matrix(Y, y_pred)
print(conf_mat)

# overall TP, FP, TN, FN values, for binary values only, what is tp and tn?
print()
tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()
print("TP:",tp)
print("FP:",fp)
print("TN:",tn)
print("FN:",fn)

## Confusion Matrix (Method 1 - Hold Out)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# sklearn hold out
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

# confusion matrix
from sklearn.metrics import confusion_matrix
# SK learn Models-> https://scikit-learn.org/stable/
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

filename = 'heart_disease_health_indicators.csv'
colNames = ['highBp','cholCheck','bmi','smoker','stroke','physActivity','fruits','hvyAlcoholConsump','anyHealthcare','sex']

data = pd.read_csv(filename, names=colNames)
array = data.values
X = array[:,0:9]
Y = array[:,9]
# Set the siz of the training and test set (in percentage)
test_size = 0.33

# random seed for the data
seed = None

# Returns 4 lists
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)

print("Naive Bayes:\n------------------------------------")
model = GaussianNB()

# train the model
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy:", round(result*100.0,2), "%")

# make predictions on unseen data
y_pred = model.predict(X_test)

#print predictions:
for prediction in y_pred:
    print("Prerdiction: ", prediction)
conf_mat = confusion_matrix(Y_test, y_pred)
print()
print(conf_mat)
# overall TP, FP, TN, FN values, for binary values only, what is tp and tn?
print()
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print("TP:",tp)
print("FP:",fp)
print("TN:",tn)
print("FN:",fn)

## Classification Report
import pandas as pd
import matplotlib.pyplot as plt

# sklearn 10FCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# confusion matrix
from sklearn.metrics import classification_report

# SK learn Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
filename = 'heart_disease_health_indicators.csv'
colNames = ['highBp','cholCheck','bmi','smoker','stroke','physActivity','fruits','hvyAlcoholConsump','anyHealthcare','sex']

data = pd.read_csv(filename, names=colNames)
array = data.values
X = array[:,0:9]
Y = array[:,9]

# Folds and seed
num_folds = 10
seed = None
print("Naive Bayes:\n------------------------------------")
kfold = KFold(n_splits=num_folds, random_state=seed)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:", round(results.mean()*100.0,2),
" Standard Deviation", round(results.std()*100.0,2))
print("")

# over all classification report
y_pred = cross_val_predict(model, X, Y, cv=10)
report = classification_report(Y, y_pred)
print(report)
