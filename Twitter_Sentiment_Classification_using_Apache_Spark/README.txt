Please download test and train data from the link given below:

http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

Format 
Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)

The data used for this project was slightly modified. The entries with 
polarity value = 2 are removed and the tweets with polarity value = 4 are
converted to value = 1 and then processed

Go through the document
https://raghu4690.bitbucket.io/dataScience/Twitter_Sentiment_Classification_using_Apache_Spark/
to find detailed explanation of the project.