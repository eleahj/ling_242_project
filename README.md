# Linguistic Analysis on Song Lyrics (Ling 242 Project)
lingAnalysisProjFinal.ipynb contains my detailed procedure for building this project. 

## PART 1. Use WordCloud to analyze Song Lyrics of Billboard Year-End Hot 100 (1965-2015) 

The dataset I am using for this is billboard_lyrics_1964-2015.csv and it is from: https://github.com/walkerkq/musiclyrics

import: 
- nltk 
- wordcloud
- pandas

### Brief Description of  Step 1:
- Preprocessing: Tokenize the lyrics from billboard_lyrics_1964-2015.csv
- POS tagging: Tag the tokens 
- Remove useless words:
-   the words from english stop wordset of nltk
-   My own words that I thought were useless such as "would", "could", "one", "may", "im", "ive", "dont", "youre", "oh", "cause", "yeah", "yeah yeah", "oh oh", "da da", "wan na", "im gon"
-     notably, I included "cause" here because it was used as "because" and was sometimes classified as "NN". 
-     I also wanted only nouns, adjectives, and adverbs.
- So after filtering, I ended up extracting the words that had tags such as 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'RBR', 'RBS' and were not stop words. 

- With WordCloud, I find the most freqeuntly used words (ex. top 10 most used words). I would call the song as a "trendy song" if it contains such words.

### Result
I am able to find the common keywords/topics of popular songs.
<img width="485" alt="Screen Shot 2021-12-01 at 9 21 14 PM" src="https://user-images.githubusercontent.com/72051758/144362525-a6c84d10-cd17-4ab0-a488-005f6a6bdd65.png">

The top 10 were: 'time', 'way', 'night', 'life', 'girl', 'life', 'man', 'heart', 'thing', 'baby'

They seem like reasonable results since they are common words that I would hear in everyday songs (subjective).


## PART 2. Generate models to predict if a lyrics would be "trendy," aka include the top 10 keywords

import: fasttext

Train with the "trendy song" data and test if other songs also follow the trend. 
- Also used lyrics-data.csv from https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres/version/3?select=lyrics-data.csv (NOT uploaded here bc the file is too big)
- Using the "trendy song" data from part 1, I labelled lyrics from lyrics-data.csv and billboard_lyrics_1964-2015.csv with 1 or 0. 
-   Label is 1 if it contains at least 3 top 10 keywords. Else, label is 0.
-     They look like: __label__0 or __label__1
- Perform sentence classification with FastText: Get training data and learn
-   I made model with labelled lyrics from lyrics-data.csv.
-   I made model2 with labelled lyrics from billboard_lyrics_1964-2015.csv.

### Result

I fed through some lyrics: 
- 1. "I got my peaches out in Georgia oh, yeah, shit I get my weed from California that's that shit I took my chick up to the North, yeah badass bitch I get my light right from the source, yeah yeah, that's it And I see you, the way I breathe you in, it's the texture of your skin I wanna wrap my arms around you, baby, never let you go, oh And I say, oh, there's nothing like your touch It's the way you lift me up, yeah And I'll be right here with you 'til the end" - Peaches by Justin Bieber
-   model: (('__label__0',), array([0.50001514]))
-   model2: (('__label__1',), array([0.50001323]))

- 2. "time way night life girl life man heart thing baby time way night life girl life man heart thing baby time way night life girl life man heart thing baby time way night life girl life man heart thing baby" -(a song from one of the datasets)
-   model: (('__label__0',), array([0.5000127]))
-   model2: (('__label__1',), array([0.50001174]))

- 3. "A Mighty Fortress is our God A Bulwark never failing Our helper He, amid the flood Of mortal ills prevailing: For still our ancient foe Doth seek to work us woe; His craft and power are great, And, armed with cruel hate, On earth is not his equal." - A Mighty Fortress (a hymn)
- model: (('__label__0',), array([0.5000124]))
- model2: (('__label__1',), array([0.50001091]))

I was able to predict the 0 or 1 label of some lyrics. However, they are not accurate because "model" seems more likely to give prediction of __label__0 and "model2" seems more likely to give prediction of __label__1. 


## 3 Future Work and Reflection

### Things to do differently if I am to do re-do this project
- extract more keywords:
-   Previously, I have extracted only top 10 key words  using wordcloud. However, increasing the number of key words that represent the top Billboard songs to 50, or maybe even more, will help make a more accurate prediction.
-   Get test data file and evaluate the models for better prediction.

### 
-   Maybe be able to generate new lyrics with these popular keywords using Markov Chain and graphs.
