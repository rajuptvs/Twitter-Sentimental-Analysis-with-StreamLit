import tweepy as tw
import streamlit as st
import pandas as pd
#from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from wordcloud import ImageColorGenerator

consumer_key = "RiOXO1ftjhag0RlYlc8SSNjcC"
consumer_key_secret = "sHm12PWU1UEs5bfzYi0yn3tK0PpYVyaSloyjeRFQaCQPWaeAx5"
access_token = "1518770620934152192-YQiBTvsrVL086vVDKs4deypOalvKiB"
access_token_secret = "dV8d4XgQu60r9FbLjy3CibFsCWwjVJy7lO3JHJQfQ9Slg"
Bearer_Token="AAAAAAAAAAAAAAAAAAAAAE8pcAEAAAAA60TYToxJMKE%2FH7kx81ACPN3JM0o%3DwUZRWCXng4z6y6u1B5qKGVLq1uIcc3gZNlYT2hKwJZLXzstN22"
auth = tw.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
#classifier = pipeline('sentiment-analysis')
st.title('Getting tweets from twitter using Twitter API and tweepy')
st.markdown('Sentiment Analysis')
def run():
    with st.form(key='Enter name'):
        search_words = st.text_input('Enter a particular topic you want to retrieve tweets for ...')
        number_of_tweets = st.number_input('Total Tweets Required (Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        tweets =tw.Cursor(api.search_tweets,q=search_words,lang='en').items(number_of_tweets)
        print(tweets)

        tweet_list = [i.text for i in tweets]
        col_name=['Latest ' +str(number_of_tweets)+' Tweets']
        df = pd.DataFrame(list(zip(tweet_list)),columns =col_name)  

        st.write(df)
        st.write(col_name)
        
        text = " ".join(i for i in df['Latest ' +str(number_of_tweets)+' Tweets'])
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)


if __name__=='__main__':
    run()


