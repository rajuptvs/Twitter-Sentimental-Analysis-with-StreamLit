# Twitter-Sentimental-Analysis-with-StreamLit

Twitter Real-Time Sentimental Analysis:
1) Twitter Data has been collected while training using Twitter API and Tweepy Module
2) Multiple Pre Processing Steps have been applied to clean the data and make the data more readable by the model.
3) Multiple Models have been used. These include :
   a) LogisticRegression
   b) DecisionTreeClassifier
   c) RandomForestClassifier
   d) Bernoulli Naive Bayes
   e) BERT model
4) Pickle Module has been used to save and load the trained model.
5) Used StreamLit for the deployment of the model and hosted it on GitHub.
6) Link to the hosted WebApp is https://share.streamlit.io/rajuptvs/twitter-sentimental-analysis-with-streamlit/main/Streamlit_v1.py
![image](https://github.com/rajuptvs/Twitter-Sentimental-Analysis-with-StreamLit/blob/main/images/streamlit_home.png)
![image](https://github.com/rajuptvs/Twitter-Sentimental-Analysis-with-StreamLit/blob/main/images/streamlit_results.png)

## Best model, that we were able to get from the above models were LogisticRegression and the BERT Model but BERT model had a very high validation loss and lower accuracy probably because of the smaller dataset hence proceeded with the deployment of LogisticRegression.
![image](https://github.com/rajuptvs/Twitter-Sentimental-Analysis-with-StreamLit/blob/main/images/confusionmatrix.jpg)
![image](https://github.com/rajuptvs/Twitter-Sentimental-Analysis-with-StreamLit/blob/main/images/Eval.jpg)




