
# coding: utf-8

# In[1]:

import csv
import re
import string
import shutil
from operator import add
from __future__ import print_function
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

listOfAllTrainingTweets = []
stopWords = stopwords.words('english')
snowBallStemmer = SnowballStemmer("english")


# In[2]:

def cleanTweets(tweetText):
    cleanedTokens = []
    listOfAllTrainingTweets.append(tweetText)
    tweetText = tweetText.strip('\n ')
    tweetText = tweetText.lower().replace('\n',' ')
    tweetText = re.sub(r'https?:\/\/.[^\s]+', 'URL', tweetText)
    tweetText = re.sub(r'www\.[^\s]+', 'URL', tweetText)
    tweetText = re.sub(r'<[A-Za-z]*>', '', tweetText)
    tweetText = re.sub(r'<\/[A-Za-z]*>', '', tweetText)
    tweetText = re.sub(r'@[^\s]+', 'AT_USER', tweetText)
    tweetText = re.sub(r'AT_USER', '', tweetText)
    tweetText = re.sub(r'URL', '', tweetText)
    tweetText = re.sub(r'\"?b[^A-Za-z]', '', tweetText)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tweetText = regex.sub('', tweetText)
    tweetText = re.sub(r'(.)\1{2,}', r'\1\1', tweetText)
    tokens = tweetText.split()
    for token in tokens:
        token = token.strip('"')
        cleanedTokens.append(snowBallStemmer.stem(token))
    #cleanedTokens = list(set(tweetText.split()))
    cleanedTokens = tweetText.split()
    for stopWord in stopWords:
        if stopWord in cleanedTokens:
            cleanedTokens.remove(stopWord)
    #print(cleanedTokens)
    return cleanedTokens


# In[3]:

# acro_dic dictionary
acro_dic = {}
# Open the file in Universal mode
with open('/home/hadoop/data/acronyms.csv', 'rU') as f:
    # Get the CSV reader and skip header
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # First column is the key, the rest is value
        acro_dic[row[0]] = row[1:]


# In[4]:

def cleanTweets1(tweetText):
    cleanedTokens = []
    listOfAllTrainingTweets.append(tweetText)
    tweetText = tweetText.strip('\n ')
    tweetText = tweetText.lower().replace('\n',' ')
    tweetText = re.sub(r'https?:\/\/.[^\s]+', 'URL', tweetText)
    tweetText = re.sub(r'www\.[^\s]+', 'URL', tweetText)
    tweetText = re.sub(r'<[A-Za-z]*>', '', tweetText)
    tweetText = re.sub(r'<\/[A-Za-z]*>', '', tweetText)
    tweetText = re.sub(r'@[^\s]+', 'AT_USER', tweetText)
    tweetText = re.sub(r'AT_USER', '', tweetText)
    tweetText = re.sub(r'URL', '', tweetText)
    tweetText = re.sub(r'\"?b[^A-Za-z]', '', tweetText)
    tweetText = re.sub(r'(.)\1{2,}', r'\1\1', tweetText)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = tweetText.split()
    for token in tokens:
        if token in acro_dic:
            tokens.extend(acro_dic[token][0].lower().split())
            tokens.remove(token)
            continue
        token = regex.sub('', token)
        cleanedTokens.append(PorterStemmer().stem(token))
    for token in cleanedTokens:
        if token in stopWords:
            cleanedTokens.remove(token)
    return cleanedTokens


# In[5]:

def parseForSkLearn(filename):
    features = []
    labels = []
    vocabulary = []
    with open(filename, newline='') as csvfile:
        for row in csvfile:
            features.append(' '. join(cleanTweets1(row.split(',', 1)[1])))
            vocabulary.extend(cleanTweets1(row.split(',', 1)[1]))
            labels.append(float(row.split(',', 1)[0]))
    return [labels, features, vocabulary]


# In[6]:

labelsObama, featuresObama, vocabObama = parseForSkLearn('/home/hadoop/data/obama.csv')
labelsRomney, featuresRomney, vocabRomney = parseForSkLearn('/home/hadoop/data/romney.csv')

labelsObamaTest, featuresObamaTest, vocabObamaTest = parseForSkLearn('/home/hadoop/data/obama_test.csv')
labelsRomneyTest, featuresRomneyTest, vocabRomneyTest = parseForSkLearn('/home/hadoop/data/romney_test.csv')
#print(featuresObama)


# In[111]:

#   print(featuresObama)


# In[10]:

import numpy as np
from sklearn import metrics

def mesaureModelPerformanceSklearn(label, pred):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(label, pred)
    accuracy = metrics.accuracy_score(label, pred)
    return [precision[0], precision[2], recall[0], recall[2], fscore[0], fscore[2], accuracy]
    


# In[11]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def performKFoldTestingTFIDF(vocabulary, features, labels, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    y = np.array(labels)
    count_vectorizer = CountVectorizer(min_df=1.0,vocabulary=set(vocabulary),ngram_range=(1, 3))
    count_vectorizer.fit(features)
    tfidf = TfidfTransformer(norm="l2")

    for i in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split( features, y, test_size=0.10, random_state=42)
        print("Without over sampling X_train length=" + str(len(X_train)))
        freq_term_matrix = count_vectorizer.transform(X_train)
        #print(freq_term_matrix.shape)
        tfidf.fit(freq_term_matrix)
        tf_idf_matrix = tfidf.transform(freq_term_matrix)
        freq_term_matrix_test = count_vectorizer.transform(X_test)
        tfidf.fit(freq_term_matrix_test)
        tf_idf_matrix_test = tfidf.transform(freq_term_matrix_test)
        model.fit(tf_idf_matrix.toarray(), y_train)
        y_pred = model.predict(tf_idf_matrix_test.toarray())
        nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
        totalNPrecision += nprecision
        totalPPrecision += pprecision
        totalNRecall += nrecall
        totalPRecall += precall
        totalNFscore += nfscore
        totalPFscore += pfscore
        totalAccuracy += accuracy
    print(dataName + " data metrics")
    print (modelName + " classifier avg positive class precision:" + str((totalPPrecision/10) * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str((totalPRecall/10) * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str((totalPFscore/10) * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str((totalNPrecision/10) * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str((totalNRecall/10) * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str((totalNFscore/10) * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str((totalAccuracy/10) * 100) + "%")


# In[69]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.over_sampling import RandomOverSampler

def performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, modelName, model, dataName):
    #print ("features=" + str(features))
    #print ("labels=" + str(labels))
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    y = np.array(labels)
    count_vectorizer = CountVectorizer(min_df=1, vocabulary=set(vocabulary), ngram_range=(1, 3))
    count_vectorizer.fit(features)
    tfidf = TfidfTransformer(norm="l2")
    ros = RandomOverSampler()

    for i in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.10, random_state=42)
        #tranform train data
        freq_term_matrix_train = count_vectorizer.transform(X_train)
        tfidf.fit(freq_term_matrix_train)
        tf_idf_matrix = tfidf.transform(freq_term_matrix_train)
        X_train_resampled ,y_train_resampled = ros.fit_sample(tf_idf_matrix.toarray(), y_train)
        print("With Over sampling : X_train length=" + str(len(X_train_resampled)))
        #transform test data
        freq_term_matrix_test = count_vectorizer.transform(X_test)
        tfidf.fit(freq_term_matrix_test)
        tf_idf_matrix_test = tfidf.transform(freq_term_matrix_test)
        #fit the model with transformed train data
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(tf_idf_matrix_test.toarray())
        nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
        totalNPrecision += nprecision
        totalPPrecision += pprecision
        totalNRecall += nrecall
        totalPRecall += precall
        totalNFscore += nfscore
        totalPFscore += pfscore
        totalAccuracy += accuracy
    print(dataName + " data metrics")
    print (modelName + " classifier avg positive class precision:" + str((totalPPrecision/10) * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str((totalPRecall/10) * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str((totalPFscore/10) * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str((totalNPrecision/10) * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str((totalNRecall/10) * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str((totalNFscore/10) * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str((totalAccuracy/10) * 100) + "%")


# In[97]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def transformAndPredictTFIDF(vocabulary, features, labels, contestant):
    performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "NB", MultinomialNB(), contestant)
    performKFoldTestingTFIDF(vocabulary,  features, labels, "NB", MultinomialNB(), contestant)
    performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "LinearSVC", LinearSVC(multi_class='ovr'), contestant)    
    performKFoldTestingTFIDF(vocabulary,  features, labels, "LinearSVC", LinearSVC(multi_class='ovr'), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary,  features, labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary,  features, labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary,  features, labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "RandomForest", RandomForestClassifier(n_estimators=50), contestant)    
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "RandomForest", RandomForestClassifier(n_estimators=50), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "XGBClassifier", XGBClassifier(), contestant)    
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "XGBClassifier", XGBClassifier(), contestant)


# In[98]:

transformAndPredictTFIDF(vocabObama, featuresObama, labelsObama, "Obama")
transformAndPredictTFIDF(vocabRomney, featuresRomney, labelsRomney, "Romney")


# In[26]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def performKFoldTestingTFIDF_Test(vocabulary, features, labels,vocabulary_test, features_test, labels_test ,modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    y = np.array(labels)

    X_train = features
    y_train = y
    X_test = features_test
    y_test = np.array(labels_test)
    
    count_vectorizer = CountVectorizer(min_df=1.0,vocabulary=set(vocabulary),ngram_range=(1, 3))
    tfidf = TfidfTransformer(norm="l2")
    freq_term_matrix = count_vectorizer.fit_transform(X_train)
    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
    
    freq_term_matrix_test = count_vectorizer.transform(X_test)
    tf_idf_matrix_test = tfidf.transform(freq_term_matrix_test)
    
    model.fit(tf_idf_matrix.toarray(), y_train)

    
    y_pred = model.predict(tf_idf_matrix_test.toarray())
    nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
    totalNPrecision += nprecision
    totalPPrecision += pprecision
    totalNRecall += nrecall
    totalPRecall += precall
    totalNFscore += nfscore
    totalPFscore += pfscore
    totalAccuracy += accuracy

    print(dataName + "Test data metrics without oversampling")
    print (modelName + " classifier avg positive class precision:" + str(totalPPrecision * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str(totalPRecall * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str(totalPFscore * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str(totalNPrecision * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str(totalNRecall * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str(totalNFscore * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str(totalAccuracy * 100) + "%")


# In[58]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.over_sampling import RandomOverSampler

def performKFoldWithOverSamplingTFIDF_Test(vocabulary, features, labels,vocabulary_test, features_test, labels_test ,modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    y = np.array(labels)
    
    X_train = features
    y_train = y
    X_test = features_test
    y_test = np.array(labels_test)
    
    count_vectorizer = CountVectorizer(min_df=1, vocabulary=set(vocabulary), ngram_range=(1, 3))
    freq_term_matrix = count_vectorizer.fit_transform(X_train)
    tfidf = TfidfTransformer(norm="l2")
    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
    
    ros = RandomOverSampler()
    
    X_train_resampled ,y_train_resampled = ros.fit_sample(tf_idf_matrix.toarray(), y_train)
    
    freq_term_matrix_test = count_vectorizer.transform(X_test)
    tf_idf_matrix_test = tfidf.transform(freq_term_matrix_test)
    
    #fit the model with transformed train data
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(tf_idf_matrix_test.toarray())
    nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
    totalNPrecision += nprecision
    totalPPrecision += pprecision
    totalNRecall += nrecall
    totalPRecall += precall
    totalNFscore += nfscore
    totalPFscore += pfscore
    totalAccuracy += accuracy
    print(dataName + " Test data metrics with oversampling")
    print (modelName + " classifier avg positive class precision:" + str(totalPPrecision * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str(totalPRecall * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str(totalPFscore * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str(totalNPrecision * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str(totalNRecall * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str(totalNFscore * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str(totalAccuracy * 100) + "%")


# In[59]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def transformAndPredictTFIDF_Test(vocabulary, features, labels,vocabulary_test, features_test, labels_test, contestant):
    performKFoldTestingTFIDF_Test(vocabulary,  features, labels, vocabulary_test, features_test, labels_test, "NB", MultinomialNB(), contestant)
    performKFoldWithOverSamplingTFIDF_Test(vocabulary,  features, labels, vocabulary_test, features_test, labels_test, "NB", MultinomialNB(), contestant)
    performKFoldTestingTFIDF_Test(vocabulary,  features, labels, vocabulary_test, features_test, labels_test, "LinearSVC", LinearSVC(C=1.8, multi_class='ovr'), contestant)
    performKFoldWithOverSamplingTFIDF_Test(vocabulary,  features, labels, vocabulary_test, features_test, labels_test, "LinearSVC", LinearSVC(C=1.8, multi_class='ovr'), contestant)
    performKFoldTestingTFIDF_Test(vocabulary,  features, labels, vocabulary_test, features_test, labels_test, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
    performKFoldWithOverSamplingTFIDF_Test(vocabulary,  features, labels, vocabulary_test, features_test, labels_test, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
    #performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "LinearSVC", LinearSVC(multi_class='ovr'), contestant)    
    #performKFoldTestingTFIDF(vocabulary,  features, labels, "LinearSVC", LinearSVC(multi_class='ovr'), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary,  features, labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary,  features, labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary,  features, labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "RandomForest", RandomForestClassifier(n_estimators=50), contestant)    
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "RandomForest", RandomForestClassifier(n_estimators=50), contestant)
#     performKFoldWithOverSamplingTFIDF(vocabulary, features, labels, "XGBClassifier", XGBClassifier(), contestant)    
#     performKFoldTestingTFIDF(vocabulary,  features, labels, "XGBClassifier", XGBClassifier(), contestant)


# In[60]:

transformAndPredictTFIDF_Test(vocabObama, featuresObama, labelsObama,vocabObamaTest, featuresObamaTest, labelsObamaTest, "Obama")
transformAndPredictTFIDF_Test(vocabRomney, featuresRomney, labelsRomney, vocabRomneyTest, featuresRomneyTest, labelsRomneyTest,"Romney")


# In[ ]:

import numpy as np
from sklearn.model_selection import KFold

def performKFoldWithCountVectorizer(features, labels, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(features)
    #X = np.array(features)
    y = np.array(labels)
    for train_index, test_index in kf.split(features):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
        totalNPrecision += nprecision
        totalPPrecision += pprecision
        totalNRecall += nrecall
        totalPRecall += precall
        totalNFscore += nfscore
        totalPFscore += pfscore
        totalAccuracy += accuracy
    print(dataName + " data metrics without oversampling")
    print (modelName + " classifier avg positive class precision:" + str((totalPPrecision/10) * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str((totalPRecall/10) * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str((totalPFscore/10) * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str((totalNPrecision/10) * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str((totalNRecall/10) * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str((totalNFscore/10) * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str((totalAccuracy/10) * 100) + "%")


# In[ ]:

import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def performKFoldWithOverSamplingWithCountVectorizer(features, labels, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    kf = KFold(n_splits=10, shuffle=True)
    
    # Apply the random over-sampling
    ros = RandomOverSampler()
    kf.get_n_splits(features)
    
    y = np.array(labels)
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index] 
        print("TRAIN:", len(X_train), "TEST:", len(X_test))
        y_train, y_test = y[train_index], y[test_index]
        X_resampled ,y_resampled = ros.fit_sample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        #print(y_pred)
        nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
        totalNPrecision += nprecision
        totalPPrecision += pprecision
        totalNRecall += nrecall
        totalPRecall += precall
        totalNFscore += nfscore
        totalPFscore += pfscore
        totalAccuracy += accuracy
    print(dataName + " data metrics with oversampling")
    print (modelName + " classifier avg positive class precision:" + str((totalPPrecision/10) * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str((totalPRecall/10) * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str((totalPFscore/10) * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str((totalNPrecision/10) * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str((totalNRecall/10) * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str((totalNFscore/10) * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str((totalAccuracy/10) * 100) + "%")


# In[ ]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def transformAndPredictWithCountVectorizer(vocabulary, features, labels, contestant):
    count_vectorizer = CountVectorizer(min_df=1.0, vocabulary=set(vocabulary), ngram_range=(1, 3))
    count_vectorizer.fit(features);
    freq_term_matrix = count_vectorizer.transform(features)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "NB", MultinomialNB(), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
    performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "Randomforest", RandomForestClassifier(n_estimators=50), contestant)
    performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "XGBClassifier", XGBClassifier(), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "NB", MultinomialNB(), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
    performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "Randomforest", RandomForestClassifier(n_estimators=50), contestant)
    performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "XGBClassifier", XGBClassifier(), contestant)


# In[ ]:

transformAndPredictWithCountVectorizer(vocabObama, featuresObama, labelsObama, "Obama")
transformAndPredictWithCountVectorizer(vocabRomney, featuresRomney, labelsRomney, "Romney")


# In[30]:

import numpy as np
from sklearn.model_selection import KFold

def performKFoldWithCountVectorizer_Test(features, labels, features_test, labels_test, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(features)
    y = np.array(labels)
    
    X_train = features
    y_train = y
    
    X_test = features_test
    y_test = np.array(labels_test) 
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
    totalNPrecision += nprecision
    totalPPrecision += pprecision
    totalNRecall += nrecall
    totalPRecall += precall
    totalNFscore += nfscore
    totalPFscore += pfscore
    totalAccuracy += accuracy
    
    print(dataName + " Test data metrics without oversampling")
    print (modelName + " classifier avg positive class precision:" + str(totalPPrecision * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str(totalPRecall* 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str(totalPFscore * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str(totalNPrecision * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str(totalNRecall * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str(totalNFscore * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str(totalAccuracy * 100) + "%")


# In[55]:

import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def performKFoldWithOverSamplingWithCountVectorizer_Test(features, labels,features_test, labels_test, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
       
    # Apply the random over-sampling
    ros = RandomOverSampler()
    X_train = features
    X_test = features_test
    y_train =  np.array(labels)
    y_test = np.array(labels_test)
    
    X_resampled ,y_resampled = ros.fit_sample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
    totalNPrecision += nprecision
    totalPPrecision += pprecision
    totalNRecall += nrecall
    totalPRecall += precall
    totalNFscore += nfscore
    totalPFscore += pfscore
    totalAccuracy += accuracy
    print(dataName + " Test data metrics with oversampling")
    print (modelName + " classifier avg positive class precision:" + str(totalPPrecision * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str(totalPRecall * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str(totalPFscore * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str(totalNPrecision * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str(totalNRecall * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str(totalNFscore * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str(totalAccuracy * 100) + "%")
    


# In[56]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def transformAndPredictWithCountVectorizer_Test(vocabulary, features, labels,vocabulary_test, features_test, labels_test, contestant):
    count_vectorizer = CountVectorizer(min_df=1.0, vocabulary=set(vocabulary), ngram_range=(1, 3))
    count_vectorizer.fit(features);
    
    freq_term_matrix = count_vectorizer.transform(features)
    freq_term_matrix_test = count_vectorizer.transform(features_test)
    
    performKFoldWithCountVectorizer_Test(freq_term_matrix.toarray(), labels, freq_term_matrix_test.toarray(),labels_test,"NB", MultinomialNB(), contestant)
    performKFoldWithOverSamplingWithCountVectorizer_Test(freq_term_matrix.toarray(), labels, freq_term_matrix_test.toarray(),labels_test,"NB", MultinomialNB(), contestant)
    performKFoldWithCountVectorizer_Test(freq_term_matrix.toarray(), labels, freq_term_matrix_test.toarray(),labels_test,"LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
    performKFoldWithOverSamplingWithCountVectorizer_Test(freq_term_matrix.toarray(), labels, freq_term_matrix_test.toarray(),labels_test,"LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "NB", MultinomialNB(), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
    #performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "Randomforest", RandomForestClassifier(n_estimators=50), contestant)
    #performKFoldWithCountVectorizer(freq_term_matrix.toarray(), labels, "XGBClassifier", XGBClassifier(), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "NB", MultinomialNB(), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "LogisticRegression", LogisticRegression(solver='sag'), contestant)
    #performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "Randomforest", RandomForestClassifier(n_estimators=50), contestant)
    #performKFoldWithOverSamplingWithCountVectorizer(freq_term_matrix.toarray(), labels, "XGBClassifier", XGBClassifier(), contestant)


# In[ ]:

transformAndPredictWithCountVectorizer_Test(vocabObama, featuresObama, labelsObama,vocabObamaTest,featuresObamaTest,labelsObamaTest, "Obama")
transformAndPredictWithCountVectorizer_Test(vocabRomney, featuresRomney, labelsRomney, vocabRomneyTest,featuresRomneyTest,labelsRomneyTest, "Romney")


# In[ ]:

import numpy as np
from sklearn.model_selection import KFold

def performKFoldTempTestingHashVectorizer(features, labels, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(features)
    #X = np.array(features)
    y = np.array(labels)
    for train_index, test_index in kf.split(features):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
        totalNPrecision += nprecision
        totalPPrecision += pprecision
        totalNRecall += nrecall
        totalPRecall += precall
        totalNFscore += nfscore
        totalPFscore += pfscore
        totalAccuracy += accuracy
    print(dataName + " data metrics")
    print (modelName + " classifier avg positive class precision:" + str((totalPPrecision/10) * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str((totalPRecall/10) * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str((totalPFscore/10) * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str((totalNPrecision/10) * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str((totalNRecall/10) * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str((totalNFscore/10) * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str((totalAccuracy/10) * 100) + "%")


# In[ ]:

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer

def transformAndPredictHashVectorizer(vocabulary, features, labels, contestant):
    hash_vectorizer = HashingVectorizer(ngram_range=(1, 3))
    hash_matrix = hash_vectorizer.fit_transform(features)    
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "KNN", KNeighborsClassifier(), contestant)
    performKFoldTempTestingHashVectorizer(hash_matrix, labels, "RandomForest", RandomForestClassifier(n_estimators=50), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "XGBClassifier", XGBClassifier(), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "LogisticRegression", LogisticRegression(), contestant)    


# In[ ]:

transformAndPredictHashVectorizer(vocabObama, featuresObama, labelsObama, "Obama")
transformAndPredictHashVectorizer(vocabRomney, featuresRomney, labelsRomney, "Romney")


# In[12]:

import numpy as np
from sklearn.model_selection import KFold

def performKFoldTempTestingHashVectorizer_Test(features, labels,features_test, labels_test, modelName, model, dataName):
    totalNPrecision = 0
    totalPPrecision = 0
    totalNRecall = 0
    totalPRecall = 0
    totalNFscore = 0
    totalPFscore = 0
    totalAccuracy = 0
    X_train = features
    X_test = features_test
    y_train = np.array(labels)
    y_test = np.array(labels_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
    totalNPrecision += nprecision
    totalPPrecision += pprecision
    totalNRecall += nrecall
    totalPRecall += precall
    totalNFscore += nfscore
    totalPFscore += pfscore
    totalAccuracy += accuracy
    print(dataName + " test data metrics")
    print (modelName + " classifier avg positive class precision:" + str(totalPPrecision * 100) + "%")
    print (modelName + " classifier avg positive class recall:" + str(totalPRecall * 100) + "%")
    print (modelName + " classifier avg positive class fscore:" + str(totalPFscore * 100) + "%")
    print (modelName + " classifier avg negative class precision:" + str(totalNPrecision * 100) + "%")
    print (modelName + " classifier avg negative class recall:" + str(totalNRecall * 100) + "%")
    print (modelName + " classifier avg negative class fscore:" + str(totalNFscore * 100) + "%")
    print (modelName + " classifier avg accuracy:" + str(totalAccuracy * 100) + "%")
    
    
#     for train_index, test_index in kf.split(features):
#         print("TRAIN:", len(train_index), "TEST:", len(test_index))
#         X_train, X_test = features[train_index], features[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         nprecision, pprecision, nrecall, precall, nfscore, pfscore, accuracy = mesaureModelPerformanceSklearn(y_test, y_pred)
#         totalNPrecision += nprecision
#         totalPPrecision += pprecision
#         totalNRecall += nrecall
#         totalPRecall += precall
#         totalNFscore += nfscore
#         totalPFscore += pfscore
#         totalAccuracy += accuracy
#     print(dataName + " test data metrics")
#     print (modelName + " classifier avg positive class precision:" + str((totalPPrecision/10) * 100) + "%")
#     print (modelName + " classifier avg positive class recall:" + str((totalPRecall/10) * 100) + "%")
#     print (modelName + " classifier avg positive class fscore:" + str((totalPFscore/10) * 100) + "%")
#     print (modelName + " classifier avg negative class precision:" + str((totalNPrecision/10) * 100) + "%")
#     print (modelName + " classifier avg negative class recall:" + str((totalNRecall/10) * 100) + "%")
#     print (modelName + " classifier avg negative class fscore:" + str((totalNFscore/10) * 100) + "%")
#     print (modelName + " classifier avg accuracy:" + str((totalAccuracy/10) * 100) + "%")


# In[13]:

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer

def transformAndPredictHashVectorizer_Test(vocabulary, features, labels, vocabulary_test, features_test, labels_test, contestant):
    hash_vectorizer = HashingVectorizer(ngram_range=(1, 4))
    hash_matrix = hash_vectorizer.fit_transform(features)   
    hash_matrix_test = hash_vectorizer.transform(features_test)
    performKFoldTempTestingHashVectorizer_Test(hash_matrix, labels,hash_matrix_test, labels_test, "LinearSVC", LinearSVC(multi_class='ovr', max_iter=1000), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "DT", DecisionTreeClassifier(random_state=0), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "KNN", KNeighborsClassifier(), contestant)
#     performKFoldTempTestingHashVectorizer_Test(hash_matrix, labels, hash_matrix_test, labels_test, "RandomForest", RandomForestClassifier(n_estimators=50), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "XGBClassifier", XGBClassifier(), contestant)
#     performKFoldTempTestingHashVectorizer(hash_matrix, labels, "LogisticRegression", LogisticRegression(), contestant)    


# In[14]:

transformAndPredictHashVectorizer_Test(vocabObama, featuresObama, labelsObama,vocabObamaTest,featuresObamaTest,labelsObamaTest, "Obama")
transformAndPredictHashVectorizer_Test(vocabRomney, featuresRomney, labelsRomney, vocabRomneyTest,featuresRomneyTest,labelsRomneyTest, "Romney")

