In this assignment I'm initially fetching some tweets from online 
and then perform following operations on the extracted tweets:
1. Compute term frequency: The term frequency is computed by using the 
   formula:
   [# of occurrences of the term in all tweets]/[# of occurrences of all terms in all
   tweets]
   Check computeFrequency explanation in the file reporta.txt for more details.
2. Determine sentiment of the tweet: The sentiment of the tweet is determined by 
   first extracting terms from the tweet and then getting the sentiment score of the 
   term from AFINN-111.txt and adding the scores of all terms of the tweet.
   Refer processAfinnFile, getMultipleWordSentiments, computeScore in tweet_sentiment.py for
   code and reportb.txt for their details.
3. Determine happiest breaking bad actors: In this operation, recent 100 tweets from
   8 breaking bad actors namely (BryanCranston, aaronpaul_8, RjMitte, deanjnorris, betsy_brandt,
   mrbobodenkirk, quiethandfilms, abqjoker, mattjonesisdead, CharlesEbaker, DanielMoncada80,
   LuisMoncada77, Krystenritter) are extracted and the sentiments of their tweets are determined
   and ordered in ascending orders. 
4. Determine happiest state: The tweets fetched in the inital step will contain the information
   regarding their authors and location. This information is what exploited in this section.
   The sentiment score of each tweet is computed as mentioned in 2nd step above. The sentiment 
   scores are then grouped per state and sorted from most positive to most negative.
   
NOTE: 
a. breaking_bad_tweets.csv file contains the most recent hundred tweets of the eight 
   breaking bad actors mentioned in step 3 above. These tweets were recent 100 tweets as 
   on 28th March 2016.
b. streaming_output_full.txt file contains the tweets on which the above mentioned operations
   performed.