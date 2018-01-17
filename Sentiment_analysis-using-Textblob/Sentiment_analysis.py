import tweepy
from textblob import TextBlob
import sys
import csv

if len(sys.argv) >= 2:
	topic = sys.argv[1]
else:
	print("By default topic is Trump.")
	topic = "Trump"

consumer_key= 'CONSUMER_KEY_HERE'
consumer_secret= 'CONSUMER_SECRET_HERE'

access_token='ACCESS_TOKEN_HERE'
access_token_secret='ACCESS_TOKEN_SECRET_HERE'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth  To login via code
# OAuthHandler  method of Tweepy

auth.set_access_token(access_token, access_token_secret)

# Calling ‘auth.set_access_token’ method on the ‘auth’ variable

api = tweepy.API(auth)

#Retrieving  Tweets
public_tweets = api.search('Trump')
# Creating a public_tweet variable to store a list of tweets

# Saving each Tweet to a CSV file
with open('sentiment_tanay.csv', 'w', newline='\n') as  f:

	writer = csv.DictWriter(f, fieldnames=['Tweet', 'Sentiment'])
	writer.writeheader()
	# To print all words to terminal , we are creating a ‘for’ loop
	for tweet in public_tweets:

		# Each tweet has  a text attribute which is a string version of it so ‘tweet.text'
		text = tweet.text
		#Cleaning tweet
		
		cleanedtext = ' '.join([word for word in text.split(' ') if len(word) > 0 and word[0] != '@' and word[0] != '#' and 'http' not in word and word != 'RT'])
		
# ‘analysis’ variable to store our analysis and call Textblob with tweet.string as the only argument 
		analysis = TextBlob(cleanedtext)
#  Perform Sentiment Analysis on Tweets
		sentiment = analysis.sentiment.polarity

#Deciding the sentiment polarity threshold 		
		if sentiment >= 0:
			polarity = 'Positive'
		else:
			polarity = 'Negative'

		print(cleanedtext, polarity)

		writer.writerow({'Tweet':text, 'Sentiment':polarity})

# polarity measures how positive or negative some text is.
# Subjectivity measures how much of an opinion it is vs how factual.
