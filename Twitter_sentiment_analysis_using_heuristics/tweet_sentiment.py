import sys
import json
import collections
import re
import string

def processAfinnFile(sent_file):
    termsAndScores = {}
    with open(sent_file,'r+') as sentFile:
        for line in sentFile:
            term,score = line.split("\t")
            termsAndScores[term] = float(score)
    return termsAndScores
	
def getMultipleWordSentiments(termsAndScores):
    #Seperate handling for the two word keys in AFINN-111.txt like 
    #green wash, messing up etc
    multipleWordTerm = []
    for term,score in termsAndScores.items():
        if ' ' in term:
            multipleWordTerm.append(term)
    return multipleWordTerm
	
def processTweetFile(tweetFileName,termsAndScores):
    scoresAndTweets = {}
    multipleWordTerm = getMultipleWordSentiments(termsAndScores)
    with open(tweetFileName,'r+') as tweetFile:
        for tweet in tweetFile:
            tweetAsJson = json.loads(tweet)
            tweetText = tweetAsJson['text'].strip()
            score = computeScore(tweetText,termsAndScores,multipleWordTerm)
            tweetText = tweetText.replace('\n',' ')
            if score in scoresAndTweets:
                scoresAndTweets[score].append(tweetText)
            else:
                scoresAndTweets[score] = [tweetText]
    return scoresAndTweets
			

def computeScore(tweetText,termsAndScores,multipleWordTerm):
    score = 0
    tweetText = tweetText.strip('\n ')
    tweetText = tweetText.lower().replace('\n',' ')
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tweetText = regex.sub('', tweetText)
    for term in multipleWordTerm:
        if term in tweetText:
            score += termsAndScores[term]
            tweetText = tweetText.replace(term,"")
    tokensInTweets = tweetText.split()
    #print (tokensInTweets)
    for token in tokensInTweets:
        #token = token.strip('\'\"-,.:;!? ')
        #print(token)
        if token in termsAndScores:
            score += termsAndScores[token]
    return score
            
def main():
    termsAndScores = {}
    scoresAndTweets = {}
    sent_file = sys.argv[1]
    tweet_file = sys.argv[2]
    termsAndScores = processAfinnFile(sent_file)
    scoresAndTweets = processTweetFile(tweet_file,termsAndScores)

    #sort in decreasing order to print topmost tweets
    od = collections.OrderedDict(sorted(scoresAndTweets.items(), reverse=True))
    #print the values
    count = 0
    for key,value in od.items():
        if count <= 10:
            for val in value:
                if count < 10:
                    print(str(key)+": "+val)
                count += 1
    #sort in ascending order to print bottomost tweets
    od = collections.OrderedDict(sorted(scoresAndTweets.items()))
    #print the values
    count = 0
    bottomTen = []
    for key,value in od.items():
        if count <= 10:
            for val in value:
                if count < 10:
                    bottomTen.append(str(key)+": "+val)
                count += 1
    for entry in reversed(bottomTen):
        print(entry)

if __name__ == '__main__':
    main()
