#package installation 
install.packages("naivebayes")
install.packages("ggplot2")
install.packages("rjson")
install.packages("bigmemory")
install.packages("biganalytics")
install.packages("bigtabulate")
install.packages("stringr")
install.packages("jsonlite")
install.packages("e1071")
install.packages("caret")
install.packages("RTextTools")
install.packages("tm")
install.packages("dplyr")
install.packages("wordcloud")
install.packages("bnlearn")
install.packages("doMC")

#imports packages
library(naivebayes)
library(ggplot2)
library(rjson)
library(bigmemory)
library(biganalytics)
library(bigtabulate)
library(stringr)
library(jsonlite)
library(e1071)
library(caret)
library(RTextTools)
library(tm)
library(dplyr)
library(wordcloud)
library(bnlearn)
library(doMC)

#streams in data into file and filters out unnecessary categories
jdata <- stream_in(file("reviewsDataset.json"))
jdata$helpful <- NULL
jdata$reviewerID <- NULL
jdata$asin<-NULL
jdata$reviewerName<-NULL
jdata$summary<-NULL
jdata$reviewTime<-NULL
jdata$unixReviewTime<-NULL

#data sets used for wordcloud
odata1 <- subset(jdata, overall == '1')
odata2 <- subset(jdata, overall == '2')
odata3 <- subset(jdata, overall == '3')
odata4 <- subset(jdata, overall == '4')
odata5 <- subset(jdata, overall == '5')

#data used for naive bayes
data1 <- jdata
data1[data1=="1"]<-"Neg"
data1[data1=="2"]<-"Neg"
data1[data1=="3"]<-"Neg"
data1[data1=="4"]<-"Pos"
data1[data1=="5"]<-"Pos"

######################################################
# SVM
######################################################
dtMatrix <- create_matrix(jdata["reviewText"])

container <- create_container(dtMatrix, jdata$overall, trainSize=3719:7436, virgin=FALSE)

model <- train_model(container, "SVM", kernel = "linear", cost = 1)

predictionData <- sample(jdata$reviewText, 3718)
predMatrix <- create_matrix(predictionData, originalMatrix=dtMatrix)

#prediction
predSize = length(predictionData);
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)

#classification
results <- classify_model(predictionContainer, model)
results

#confusion matrix
jdata.test <- jdata[1:3718,]
confsvm.mat <- confusionMatrix(results$SVM_LABEL,jdata.test$overall)
confsvm.mat$overall['Accuracy']

#plots frequency of predictions
Overall <- results$SVM_LABEL
p<-ggplot(results, aes(Overall)) + geom_bar()
p

######################################################
# Naive Bayes 
######################################################
registerDoMC(cores=detectCores())
glimpse(data1)
set.seed(1)
data1$overall<-as.factor(data1$overall)

#randomize
data1 <- data1[sample(nrow(data1)), ]
data1 <- data1[sample(nrow(data1)), ]

glimpse(data1)
#tokenisation
corpus <-Corpus(VectorSource(data1$reviewText))
corpus
inspect(corpus[1:3])

#cleanup
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
dtm <- DocumentTermMatrix(corpus.clean)
inspect(dtm[40:50, 10:15])

data1.train <- data1[1:3718,]
data1.test <- data1[3719:7436,]

dtm.train <- dtm[1:3718,]
dtm.test <- dtm[3719:7436,]

corpus.clean.train <- corpus.clean[1:3718]
corpus.clean.test <- corpus.clean[3719:7436]

dim(dtm.train)

fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))

#creates document term matrix
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))

dim(dtm.train.nb)

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))

dim(dtm.train.nb)

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

#classification and prediction
system.time( classifier <- naiveBayes(trainNB, data1.train$overall, laplace = 1) )
system.time( pred <- predict(classifier, newdata=testNB) )

#creates table with predictions and actual
nbt<-table("Predictions"= pred,  "Actual" = data1.test$overall )
nbt

#consuion matrix
conf.mat <- confusionMatrix(pred, data1.test$overall)

conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']

#barplot predictions vs. actual
barplot(nbt, main= "Predictions vs. Actual", legend = (c("Actual","Predicted")) )
######################################################
# Word Cloud
######################################################
wordcloud(corpus.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

corpus <- Corpus(VectorSource(jdata$reviewText))
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
wordcloud(corpus.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

corpus1 <- Corpus(VectorSource(odata1$reviewText))
corpus1.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
wordcloud(corpus1.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

corpus2 <- Corpus(VectorSource(odata2$reviewText))
corpus2.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
wordcloud(corpus2.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

corpus3 <- Corpus(VectorSource(odata3$reviewText))
corpus3.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
wordcloud(corpus3.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

corpus4 <- Corpus(VectorSource(odata4$reviewText))
corpus4.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
wordcloud(corpus4.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

corpus5 <- Corpus(VectorSource(odata5$reviewText))
corpus5.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
wordcloud(corpus5.clean, scale=c(6,0.7), max.words=150, 
          random.order=FALSE, rot.per=0.35,colors=brewer.pal(8,"Dark2"))

######################################################
# Summarized Results
######################################################



######## SVM - Final Results and accuracy ########
confsvm.mat
confsvm.mat$overall['Accuracy']


########  NaiveBayes - Final Results ands accuracy ########
conf.mat
conf.mat$overall['Accuracy']

