import tweepy as tw
import streamlit as st
import pandas as pd
import tensorflow as tf
#from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import string
from wordcloud import ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
#import joblib
import nltk
nltk.download('stopwords')

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
model = open("LRmodel.pkl","rb")
classifier = pickle.load(model)
#m_jlib = joblib.load('model_jlib')

MAP={0:'Negative', 1:'Neutral', 2:'Positive'}

target='Sentiment'
st.markdown('Sentiment Analysis')

porter=PorterStemmer()

def preprocessor(text):
    text = text.lower()
    text = ''.join([i for i in text if i in string.ascii_lowercase+' '])
    text = ' '.join([PorterStemmer().stem(word) for word in text.split()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text



def tokenizer(text):
        return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]



tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True,max_features=50)

def run():
    #Input Fields
    with st.form(key='Enter name'):
        search_words = st.text_input('Enter a particular topic you want to retrieve tweets for ...')
        number_of_tweets = st.number_input('Total Tweets Required (Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')


    #Getting Tweets using twitter API
    if submit_button:
        tweets =tw.Cursor(api.search_tweets,q=search_words,lang='en').items(number_of_tweets)
        print(tweets)

        tweet_list = [i.text for i in tweets]
        col_name=['Latest ' +str(number_of_tweets)+' Tweets']
        df = pd.DataFrame(list(zip(tweet_list)),columns =col_name)  
        st.subheader("Below are the recent tweets related to  "+ search_words)
        st.write(df)

        #Pre Processing of the tweets
        for i in tqdm(range(df.shape[0])):

            df.loc[i,'processed_text'] = preprocessor(df['Latest ' +str(number_of_tweets)+' Tweets'][i])

        st.subheader("Below are the processed tweets related to "+ search_words)
        st.write(df)

        #Tokenization of the tweets for getting the features
        x=tfidf.fit_transform(df.processed_text)

        #Using the saved model to make the predictions on the tweets
        prediction=classifier.predict(x)

        Y = pd.Series(prediction).map(MAP)

        #Converting timeseries to DataFrame
        Y=Y.to_frame()
        df['Sentiment']=Y

        #Final DataFrame
        st.subheader("Sentimental Analysis of tweets related to "+ search_words)
        st.write(df)

        #Image Cloud Generation
        st.subheader("Most occured words in tweets related to "+ search_words)
        text = " ".join(i for i in df['processed_text'])
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)


        #st.write(filepath)


if __name__=='__main__':
    run()


