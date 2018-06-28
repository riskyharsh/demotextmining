# DEDA Demo: News Classification with Scikit-Learn

# First the Newsgroup Data is downloaded (http://scikit-learn.org):

from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all')
newsgroups.target_names

# We only consider a subset of the original data:

cats = ['talk.religion.misc', 'rec.sport.baseball', 'rec.sport.hockey']
newsgroups_subset = fetch_20newsgroups(subset='all', categories=cats, remove=('headers','footers','quotes'))

text_number = -2
print(newsgroups_subset.data[text_number])

# A vectorizer function is implemented in the Scikit-Learn package. You can directly filter out stopwords.

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(newsgroups_subset.data).toarray()
y = newsgroups_subset.target

print(type(X))
print(X.shape)

# The list of stopwords can be displayed by the following attribute:

print(vect.get_stop_words())

# In a next step the dataset is split into 90% training data and 10% test data.

n_samples = X.shape[0]
X_train = X[:int(.9*n_samples)]
y_train = y[:int(.9*n_samples)]
X_test = X[int(.9*n_samples):]
y_test = y[int(.9*n_samples):]

print(X_train.shape)

# Feature Selection

# Features are now selected according to their Chi2 value.

from sklearn.feature_selection import SelectKBest, chi2

a = 5
selector = SelectKBest(chi2, k=int(a*X_train.shape[1]**(1/2.0)))
X_train_select = selector.fit_transform(X_train, y_train)
ind = selector.get_support(indices=True)
X_test_select = X_test[:, ind]
 
print(X_train_select.shape)

# It is interesting to see which term weights are most dependent on the classes:

selector_buzzwords = SelectKBest(chi2, k=20)
selector_buzzwords.fit(X_train, y_train)
ind_buzzwords = selector_buzzwords.get_support(indices=True)

vocab_inverse = {v: k for k,v in vect.vocabulary_.items()}

for i in list(ind_buzzwords):
    print(vocab_inverse[i])

# Models

# Gaussian Naive Bayes

# We check first how well the Gaussian Naive Bayes classifier performs.

from sklearn.naive_bayes import GaussianNB

gau = GaussianNB()
print('Gaussian Naive Bayes Score: %f' % gau.fit(X_train_select, y_train).score(X_test_select, y_test))

# Since we have our first model, let us predict the class for a specific text:

print(newsgroups_subset.target_names[int(gau.predict(X_test_select[text_number].reshape([1,X_test_select.shape[1]])))])

# We can also take a look at the confusion matrix:

from sklearn import metrics

print(newsgroups_subset.target_names)
print(metrics.confusion_matrix(y_test, gau.predict(X_test_select)))

# Note that the confusion matrix C is such that Ci,j is equal to the number of observations known to be in class i but predicted to be in class j

# Logistic Regression

# Next classifier which is investigated is the Logistic Regression classifier. Various hyperparameter values of the classifier can be applied via default 3-fold cross-validation (within GridSearchSV) which works as follows:

# Split the training set again into 3 subsets (one of the subsets is will be used as an evaluation set)
# Given a specific hyperparameter set, train the model on each each of the two remaining subsets separately
# Evaluate (compute performance measures for) both models on the evaluation set
# The performance measure for the hyperparameter set (e.g. accuracy) is then calculated as the average of the two different performance measures
# The final model (and the final parameter set) is the one with the highest average performance

from sklearn import linear_model
from sklearn.grid_search import GridSearchCV

log = linear_model.LogisticRegression(solver='sag') 

parameters = {'class_weight':(None, 'balanced'), 'multi_class':('ovr', 'multinomial')}
gridsearch = GridSearchCV(log, parameters, cv=5)
LR_score = gridsearch.fit(X_train_select, y_train).score(X_test_select, y_test)
print('Logistic Regression Score: %f' % LR_score)

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Discriminant Analysis

# The Discriminant Analysis yiels the following results:

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver='lsqr')

parameters = {'shrinkage':(None, 'auto')}
gridsearch = GridSearchCV(lda, parameters)
DA_score = gridsearch.fit(X_train_select, y_train).score(X_test_select, y_test)
print('Discriminant Analysis Score: %f' % DA_score)

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Neural Network

# We now consider a Neural Network classifier:

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=2000, learning_rate='adaptive')

parameters = {'hidden_layer_sizes':((100,100,100), (100,100), (100,))}
gridsearch = GridSearchCV(mlp, parameters)
NN_score = gridsearch.fit(X_train_select, y_train).score(X_test_select, y_test)
print('Neural Network Score: %f' % NN_score)

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Decision Tree

# Now the Decision Tree classifier is assessed:

from sklearn import tree

tree = tree.DecisionTreeClassifier()

parameters = {'class_weight':(None, 'balanced'), 'criterion':('gini', 'entropy')}
gridsearch = GridSearchCV(tree, parameters)
DT_score = gridsearch.fit(X_train_select, y_train).score(X_test_select, y_test)
print('Decision Tree Score: %f' % DT_score)

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Linear Support Vector Classifier

# We turn to the Linear Support Vector classifier now:

from sklearn.svm import LinearSVC

lsvc = LinearSVC()

parameters = {'multi_class':('ovr', 'crammer_singer'), 'class_weight':(None, 'balanced')}
gridsearch = GridSearchCV(lsvc, parameters)
LSVC_score = gridsearch.fit(X_train_select, y_train).score(X_test_select, y_test)
print('Linear SVC Score: %f' % LSVC_score)

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Kernel Support Vector Classifier

# Lastly, the Kernel Support Vector classifier is considered:

from sklearn.svm import SVC

svc = SVC()

parameters = {'kernel':('rbf', 'linear')}
gridsearch = GridSearchCV(svc, parameters)
KSVC_score = gridsearch.fit(X_train_select, y_train).score(X_test_select, y_test)
print('Kernel SVC Score: %f' % KSVC_score)

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Summary: Model Scores

print('Logistic Regression: %f' % LR_score)
print('Discriminant Analysis: %f' % DA_score)
print('Neural Network: %f' % NN_score)
print('Decision Tree: %f' % DT_score)
print('Linear SVC: %f' % LSVC_score)
print('Kernel SVC: %f' % KSVC_score)