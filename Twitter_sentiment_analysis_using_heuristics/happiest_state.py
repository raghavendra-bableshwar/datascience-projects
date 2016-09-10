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

def getStateNamesWithAbbr():
    stateNamesWithAbbr = {
        "Alabama":"AL",
        "Alaska":"AK",
        "Arizona":"AZ",
        "Arkansas":"AR",
        "California":"CA",
        "Colorado":"CO",
        "Connecticut":"CT",
        "Delaware":"DE",
        "District of Columbia":"DC",
        "Florida":"FL",
        "Georgia":"GA",
        "Hawaii":"HI",
        "Idaho":"ID",
        "Illinois":"IL",
        "Indiana":"IN",
        "Iowa":"IA",
        "Kansas":"KS",
        "Kentucky":"KY",
        "Louisiana":"LA",
        "Maine":"ME",
        "Montana":"MT",
        "Nebraska":"NE",
        "Nevada":"NV",
        "New Hampshire":"NH",
        "New Jersey":"NJ",
        "New Mexico":"NM",
        "New York":"NY",
        "North Carolina":"NC",
        "North Dakota":"ND",
        "Ohio":"OH",
        "Oklahoma":"OK",
        "Oregon":"OR",
        "Maryland":"MD",
        "Massachusetts":"MA",
        "Michigan":"MI",
        "Minnesota":"MN",
        "Mississippi":"MS",
        "Missouri":"MO",
        "Pennsylvania":"PA",
        "Rhode Island":"RI",
        "South Carolina":"SC",
        "South Dakota":"SD",
        "Tennessee":"TN",
        "Texas":"TX",
        "Utah":"UT",
        "Vermont":"VT",
        "Virginia":"VA",
        "Washington":"WA",
        "West Virginia":"WV",
        "Wisconsin":"WI",
        "Wyoming":"WY"
    }
    return stateNamesWithAbbr
    
def getStateFromUser(stateNamesWithAbbr, line):
    line = line.lower()
    for state,abbr in stateNamesWithAbbr.items():
        if state.lower() in line:
            return stateNamesWithAbbr[state]
    return None


def processTweetFile(tweetFileName,termsAndScores):
    statesAndScores = {}
    statesAndItsFreq = {}
    multipleWordTerm = getMultipleWordSentiments(termsAndScores)
    stateNamesWithAbbr = getStateNamesWithAbbr()
    count = 0
    with open(tweetFileName,'r+') as tweetFile:
        for tweet in tweetFile:
            tweetAsJson = json.loads(tweet)
            state = ''
            if tweetAsJson['text'] != None and tweetAsJson['lang'] == "en":
                if tweetAsJson['place'] != None and tweetAsJson['place']['country_code'] == 'US':
                    #count+=1
                    if tweetAsJson['place']['place_type'] == 'city':
                        state = tweetAsJson['place']['full_name'].split(',')[1].strip()
                    elif tweetAsJson['place']['place_type'] == 'admin':
                        stateName = tweetAsJson['place']['full_name'].split(',')[0].strip()
                        #print (stateName)
                        if stateName in stateNamesWithAbbr:
                            state = stateNamesWithAbbr[stateName]
                elif tweetAsJson['user'] != None and tweetAsJson['user']['location'] != None:
                    #print (tweetAsJson['user']['location']+":")
                    stateName = getStateFromUser(stateNamesWithAbbr,tweetAsJson['user']['location'])
                    if stateName != None:
                        #print (tweetAsJson['user']['location']+":"+stateName)
                        #count += 1
                        state = stateName
                        #print(state)
                    #print (tweetAsJson['user']['location'])
            if state != '':
                tweetText = tweetAsJson['text']
                score = computeScore(tweetText,termsAndScores,multipleWordTerm)
                if state in statesAndScores:
                    statesAndScores[state] += score
                    statesAndItsFreq[state] += 1
                else:
                    statesAndScores[state] = score
                    statesAndItsFreq[state] = 1
    #print (count)
    for state,freq in statesAndItsFreq.items():
        statesAndScores[state] /= freq
    return statesAndScores
    
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
    statesAndScores = {}
    sent_file = sys.argv[1]
    tweet_file = sys.argv[2]
    termsAndScores = processAfinnFile(sent_file)
    statesAndScores = processTweetFile(tweet_file,termsAndScores)
    #sort in decreasing order
    statesAndScores = sorted(statesAndScores.items(), key=lambda x:x[1], reverse=True)
    for entries in statesAndScores:
        print (str(entries[1])+": "+entries[0])
    

if __name__ == '__main__':
    main()
