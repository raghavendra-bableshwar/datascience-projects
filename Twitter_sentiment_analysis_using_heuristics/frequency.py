import sys
import json
import string
import re
#from pprint import pprint

def processStopWordFile(stopWordFileName):
    stopWordList = []
    with open(stopWordFileName,'r+') as stopWordFile:
        for stopWord in stopWordFile:
            stopWordList.append(stopWord.strip())
    return stopWordList

def processTweetFile(tweetFileName):
    termsWithCount = {}
    with open(tweetFileName,'r+') as tweetFile:
        for tweet in tweetFile:
            tweetAsJson = json.loads(tweet)
            ########################################
            tweetText = tweetAsJson['text'].lower().strip('\n"')
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            tweetText = regex.sub('', tweetText)
			########################################
            #tokensInTweets = tweetAsJson['text'].lower().split()
            tokensInTweets = tweetText.split()
            if tweetAsJson['lang'] == 'en':
                for token in tokensInTweets:
                    if token not in termsWithCount:
                        termsWithCount[token] = 1
                    else:
                        termsWithCount[token] += 1
    return termsWithCount

def computeFrequency(stopWordList,termsWithCount):
    termFreqDict = {}
    #Removing stopwords in termsWithCount dictionary
    for word in stopWordList:
        if word in termsWithCount:
            del termsWithCount[word]
    #Get the number of occurances of all terms in all tweets
    allTermsCount = sum(termsWithCount.values())
    #print (allTermsCount)
    for term,count in termsWithCount.items():
        freq = count/allTermsCount
        termFreqDict[term] = freq
    return termFreqDict


def main():
    stopWordList = []
    termsWithCount = {}
    termFreqDict = {}
    stopWordFileName = sys.argv[1]
    tweetFileName = sys.argv[2]
    stopWordList = processStopWordFile(stopWordFileName)
    termsWithCount = processTweetFile(tweetFileName)
    termFreqDict = computeFrequency(stopWordList,termsWithCount)
    termFreqDict = sorted(termFreqDict.items(), key=lambda x:x[1], reverse=True)
    #termFreqDict = ((k, termFreqDict[k]) for k in sorted(termFreqDict, key=termFreqDict.get, reverse=True))
    for entries in termFreqDict:
        print (entries[0]+" "+str(entries[1]))

if __name__ == '__main__':
    main()
