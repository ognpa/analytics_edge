# KAGGLE COMPETITION - GETTING STARTED

# This script file is intended to help you get started on the Kaggle platform, and to show you how to make a submission to the competition.

library(tm)
library(e1071)
library(caTools)
library(ROCR)
library(randomForest)


eBayTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE)
eBayTrain[1861,]
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE)

eBayTest$sold<-NA
total=rbind(eBayTrain,eBayTest)


total$description<- sapply(total$description,function(row) iconv(row, "latin1", "ASCII", sub=""))
total$condition = as.factor(total$condition)
total$cellular = as.factor(total$cellular)
total$carrier = as.factor(total$carrier)
total$color = as.factor(total$color)
total$storage = as.factor(total$storage)
total$productline = as.factor(total$productline)
median_price_by_productline=aggregate(startprice~productline,data=total,median)
total$median_price=median_price_by_productline[match(total$productline,median_price_by_productline$productline),2]

total$median_price=log10(total$median_price)
total$startprice=log10(total$startprice)

total[1861,]

train=head(total,nrow(eBayTrain))
test=tail(total,nrow(eBayTest))


#Text mining
CorpusDescription = Corpus(VectorSource(c(total$description)))
CorpusDescription = tm_map(CorpusDescription, content_transformer(tolower), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, PlainTextDocument, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removePunctuation, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removeWords, stopwords("english"), lazy=TRUE)

CorpusDescription = tm_map(CorpusDescription, stemDocument, lazy=TRUE)
dtm = DocumentTermMatrix(CorpusDescription)
sparse = removeSparseTerms(dtm, 0.990)
DescriptionWords = as.data.frame(as.matrix(sparse))
colnames(DescriptionWords) = make.names(colnames(DescriptionWords))
DescriptionWordsTrain = head(DescriptionWords, nrow(train))
DescriptionWordsTest = tail(DescriptionWords, nrow(test))

#Add text features to train and test sets
ebay_train=cbind(train,DescriptionWordsTrain)
ebay_test=cbind(test,DescriptionWordsTest)

ebay_train$description=NULL
ebay_test$description=NULL
names(ebay_train) = make.names(names(ebay_train), unique=TRUE)
names(ebay_test) = make.names(names(ebay_test), unique=TRUE)

#Recursive feature selection
library(caret)
ebay_train_minus_sold=ebay_train
ebay_train_minus_sold$sold=NULL

set.seed(9232)
subsets=seq(3,80,2)


rfFuncs$summary <- twoClassSummary
trainctrl <- trainControl(classProbs= TRUE,
                          summaryFunction = twoClassSummary)
control <- rfeControl(functions=rfFuncs, method="repeatedcv", number=10, repeats = 4, verbose=TRUE)
results <- rfe(ebay_train_minus_sold, as.factor(ebay_train$sold), sizes=subsets, rfeControl=control,metric="ROC", trControl = trainctrl)

results

# Transform Dataset to remove discarded features
predictors(results)

ebay_train = ebay_train[c(predictors(results), "sold")]
ebay_test = ebay_test[c(predictors(results), "sold")]
correlationMatrix=cor(ebay_train[sapply(ebay_train, is.numeric)])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7)
highlyCorrelated
ebay_train = ebay_train[,-c(highlyCorrelated)]
ebay_test = ebay_test[,-c(highlyCorrelated)]


str(train)
spl=sample.split(ebay_train,0.7)
train=subset(ebay_train,spl==TRUE)
test=subset(ebay_train,spl==FALSE)


ebayRF =  randomForest(as.factor(sold) ~., data = train,keep.forest=TRUE, ntree = 600)
pref=predict(ebayRF,test)
table(test$sold,pref)

probs<-predict(ebayRF,ebay_test,type='prob')[,2]
MySubmission = data.frame(UniqueID = ebay_test$UniqueID, Probability1 = probs)
write.csv(MySubmission, "Submission-2.csv", row.names=FALSE)

# You should upload the submission "SubmissionSimpleLog.csv" on the Kaggle website to use this as a submission to the competition

