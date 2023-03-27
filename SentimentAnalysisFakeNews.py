import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def sentiment_scores(sentence, lstScores,totalCompoundScore, totalNegScore, totalNeuScore, totalPosScore):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    totalNegScore[0] += sentiment_dict['neg']
    totalNeuScore[0] += sentiment_dict['neu']
    totalPosScore[0] += sentiment_dict['pos']
    totalCompoundScore[0]+=sentiment_dict['compound']
    if sentiment_dict['compound'] >= 0.05:
         lstScores[0]+=1
    elif sentiment_dict['compound'] <= - 0.05:
         lstScores[1] += 1
    else:
         lstScores[2] += 1
fileName=input("Enter file name (Example: file1.csv):")
try:
    df = pd.read_csv(fileName, encoding='utf-8', encoding_errors='ignore')
except Exception as e:
    print(str(e))
lstTextF = df["text"].tolist()
lstScoresMainF=[0,0,0]
totalCompoundF=[0]
totalNegF=[0]
totalNeuF=[0]
totalPosF=[0]
print("processing...")
counter=0;
for index, item in enumerate(lstTextF):
    try:
        sentiment_scores(item, lstScoresMainF,totalCompoundF,totalNegF,totalNeuF,totalPosF)
        print("processing row no.",counter)
    except Exception as ee:
        print(str(ee))
    counter=counter+1
print("Number of scores: [Positive, Negative, Neutral]:",lstScoresMainF)
print("average Sentiment:", totalCompoundF[0]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("differentiated Sentiment:")
print("Negative average:",totalNegF[0]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("Neutral average:",totalNeuF[0]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("Positive average:",totalPosF[0]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))

