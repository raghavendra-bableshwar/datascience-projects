import sys
import csv
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
	
def processCSVFile(csv_file,termsAndScores):
    actorsAndScores = {}
    actorsAndFreq = {}
    actorsAndTweetList = []
    multipleWordTerm = getMultipleWordSentiments(termsAndScores)
    with open(csv_file,newline='') as csvFile:
        actorReader = csv.reader(csvFile, delimiter=',',dialect='excel')
        for row in actorReader:
            #print (row[0])
            #print (row[1])
            actor = row[0]
            tweetText = row[1]
            if actor == 'user_name':
                continue
            score = computeScore(tweetText,termsAndScores,multipleWordTerm)
            if actor in actorsAndScores:
                actorsAndScores[actor] += score
                actorsAndFreq[actor] += 1
            else:
                actorsAndScores[actor] = score
                actorsAndFreq[actor] = 1
            #print (row[0].split(',')[0])
    for actor,freq in actorsAndFreq.items():
        actorsAndScores[actor] /= freq
    return actorsAndScores
	

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
    actorsAndScores = {}
    sent_file = sys.argv[1]
    csv_file = sys.argv[2]
    termsAndScores = processAfinnFile(sent_file)
    actorsAndScores = processCSVFile(csv_file,termsAndScores)
    #sort in decreasing order
    actorsAndScores = sorted(actorsAndScores.items(), key=lambda x:x[1], reverse=True)
    for entries in actorsAndScores:
        print (str(entries[1])+": "+entries[0])

if __name__ == '__main__':
    main()
