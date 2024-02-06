import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
import pandas as pd
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
totalTextBlobScore=0
totalAfinnScore=0
totalTextBlobSubjectivity=0
counter=0
totalAfinnNegatives=0
totalAfinnPositives=0
totalAfinnNeutrals=0
totalTextBlobNegatives=0
totalTextBlobPositives=0
totalTextBlobNeutrals=0
print("processing...")

for index, item in enumerate(lstTextF):
    try:
        sentiment_scores(item, lstScoresMainF,totalCompoundF,totalNegF,totalNeuF,totalPosF)
        af = Afinn()
        af_scr = af.score(item)
        af_norm = float(af_scr / 5)
        if af_norm >= 0.05:
            totalAfinnPositives += 1
        elif af_norm <= - 0.05:
            totalAfinnNegatives += 1
        else:
            totalAfinnNeutrals += 1
        txtblb = TextBlob(item)
        txtblb_score = txtblb.polarity
        if txtblb_score >= 0.05:
            totalTextBlobPositives += 1
        elif txtblb_score <= - 0.05:
            totalTextBlobNegatives += 1
        else:
            totalTextBlobNeutrals += 1
        totalTextBlobScore += txtblb_score
        totalTextBlobSubjectivity += txtblb.subjectivity
        totalAfinnScore += af_norm
        counter = counter + 1
        print("processing row no.", counter)
    except Exception as ee:
        print(str(ee))

lstAfinn=[totalAfinnPositives,totalAfinnNegatives,totalAfinnNeutrals]
lstTextBlob=[totalTextBlobPositives,totalTextBlobNegatives,totalTextBlobNeutrals]
print ("For the file",fileName,"that has",counter,"records:")
print("Vader number of scores: [Positive, Negative, Neutral]:",lstScoresMainF)
print("Vader average Sentiment:", totalCompoundF[0]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("Vader negative percentage:",100* lstScoresMainF[1]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("Vader positive percentage:",100* lstScoresMainF[0]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("Vader neutral percentage:",100* lstScoresMainF[2]/(lstScoresMainF[0]+lstScoresMainF[1]+lstScoresMainF[2]))
print("Afinn average sentiment:", totalAfinnScore/counter)
print("Afinn number of scores: [Positive, Negative, Neutral]:", lstAfinn)
print("Afinn negatives percentage:",100*  totalAfinnNegatives/counter)
print("Afinn positives percentage:",100*  totalAfinnPositives/counter)
print("Afinn neutrals percentage:",100*  totalAfinnNeutrals/counter)
print("TextBlob average sentiment:", totalTextBlobScore/counter)
print("TextBlob number of scores: [Positive, Negative, Neutral]:", lstTextBlob)
print("TextBlob negatives percentage:", 100* totalTextBlobNegatives/counter)
print("TextBlob positives percentage:",100* totalTextBlobPositives/counter)
print("TextBlob neutrals percentage:",100*  totalTextBlobNeutrals/counter)
print("TextBlob average subjectivity:", totalTextBlobSubjectivity/counter)
