import argparse
import oauth2 as oauth
import urllib.request as urllib
import json
import sys
import csv

# See Assignment 1 instructions for how to get these credentials
access_token_key = "1961093533-HByGN5hPAzYyCHgwFNTMkpI9wTf0zZ0phD91nsX"
access_token_secret = "yvs0DhhA2hQ9ap42EO6NulplZ62vRWgFu75P9xgcokKoB"

consumer_key = "YOHiEKfo6Xu0MdusfnLyjzPmb"
consumer_secret = "li7lCTf41maDxBR7uS4El347eS3SLqDBAHE5XaTfVbEXeOBo03"

_debug = 0

oauth_token    = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=consumer_key, secret=consumer_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"


http_handler  = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

'''
Construct, sign, and open a twitter request
using the hard-coded credentials above.
'''
def twitterreq(url, method, parameters):
    req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url, 
                                             parameters=parameters)

    req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

    headers = req.to_header()

    if http_method == "POST":
        encoded_post_data = req.to_postdata()
    else:
        encoded_post_data = None
        url = req.to_url()

    opener = urllib.OpenerDirector()
    opener.add_handler(http_handler)
    opener.add_handler(https_handler)

    response = opener.open(url, encoded_post_data)

    return response

def fetch_samples():
    url = "https://stream.twitter.com/1.1/statuses/sample.json?language=en"
    parameters = []
    response = twitterreq(url, "GET", parameters)
    #temporary modification: Added decode to print function
    for line in response:
        print (line.strip().decode('utf-8'))

def fetch_by_terms(term):
    url = "https://api.twitter.com/1.1/search/tweets.json"
    parameters = [("q", term),("count",100)]
    response = twitterreq(url, "GET", parameters)
    print (response.readline())

def fetch_by_user_names(user_name_file):
    #TODO: Fetch the tweets by the list of usernames and write them to stdout in the CSV format
    sn_file = open(user_name_file)
    url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    #userNameWithTweets = {}
    tweetText = []
    tweetText.append('user_name,tweet'.split(',',1))
    with open(user_name_file,'r') as sn_file:
        for username in sn_file:
            username = username.strip()
            parameters = [("screen_name",username),("count",100)]
            response = twitterreq(url, "GET", parameters)
            tweets = json.loads(response.read().decode('utf-8'))
            for tweet in tweets:
                tweetText.append((username+","+tweet['text'].strip()).split(',',1))
    #writer = csv.writer(sys.stdout,quotechar='"',quoting=csv.QUOTE_MINIMAL,dialect='excel',delimiter='\n')
    writer = csv.writer(sys.stdout,dialect='excel',delimiter=',')
    writer.writerows(tweetText)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', required=True, help='Enter the command')
    parser.add_argument('-term', help='Enter the search term')
    parser.add_argument('-file', help='Enter the user name file')
    opts = parser.parse_args()
    if opts.c == "fetch_samples":
        fetch_samples()
    elif opts.c == "fetch_by_terms":
        term = opts.term
        print (term)
        fetch_by_terms(term)
    elif opts.c == "fetch_by_user_names":
        user_name_file = opts.file
        fetch_by_user_names(user_name_file)
    else:
        raise Exception("Unrecognized command")
