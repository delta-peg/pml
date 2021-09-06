library(caret); library(kernlab); data(spam)

inTrain = createDataPartition(y=spam$type, p=0.75, list = F)


training = spam[inTrain,]
testing = spam[-inTrain,]

modelFit = train(type ~., data = training, method = "glm")
##########################

library(ISLR); library(ggplot2); library(caret)
data(Wage)
summary(Wage)

inTrain = createDataPartition(y=Wage$wage, p=0.75, list = F)
training = Wage[inTrain,]
featurePlot(x=training[,c("age", "education", "jobclass")], y = training$wage, plot="pairs")

# preprocessing
##########################
library(caret); library(kernlab); data(spam)

inTrain = createDataPartition(y=spam$type, p=0.75, list = F)


training = spam[inTrain,]
testing = spam[-inTrain,]

# PCA
##########################
library(caret); library(kernlab); data(spam)

inTrain = createDataPartition(y=spam$type, p=0.75, list = F)


training = spam[inTrain,]
testing = spam[-inTrain,]

M = abs(cor(training[,-58]))
diag(M) = 0
which(M > 0.8, arr.ind = T)

smallSpam = spam[,c(34,32)]
prComp = prcomp(smallSpam)
prComp

# Quiz 2
##########################
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
testIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[-testIndex,]
testing = adData[testIndex,]

#2
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

library(ggplot2)
mixtures$index = as.numeric(row.names(mixtures))
ggplot(data=mixtures, mapping = aes(x=index, y=CompressiveStrength)) + 
  geom_point(aes(color=FlyAsh))
ggplot(data=mixtures, mapping = aes(x=index, y=CompressiveStrength)) + 
  geom_point(aes(color=Age))
#3
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

ggplot(mixtures, aes(Superplasticizer)) + geom_histogram()
ggplot(mixtures, aes(log(mixtures$Superplasticizer))) + geom_histogram()
       
#4
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

library(dplyr)
ilPredictors = training %>% as_tibble() %>% select(starts_with("IL"))
ilPredictors %>% preProcess(method = "pca", outcome = diagnosis, thresh = 0.8)

#5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
library(dplyr)
ilPredictors = training %>% 
  select(starts_with("IL") | diagnosis)

glmFit = train(diagnosis ~ ., method="glm", data=ilPredictors)

preprocessed = ilPredictors %>% select(!diagnosis) %>%  preProcess(method = "pca", thresh = 0.8)
trainPC = predict(preprocessed, ilPredictors %>% select(!diagnosis)) %>% 
  mutate(diagnosis=diagnosis[inTrain])
pcaFit = train(diagnosis ~ ., method="glm", data=trainPC)

# predict using glm
confusionMatrix(testing$diagnosis, predict(glmFit, newdata = testing))

# predict using PCA
testPC = predict(preprocessed, testing)
confusionMatrix(testing$diagnosis, predict(pcaFit, testPC))

##### week 3
#########################
# bagging = Bootstrap aggregating
data(iris)
library(ggplot2)
inTrain = createDataPartition(y=iris$Species, p=0.7, list = F)
training = iris[inTrain,]
testing = iris[-inTrain,]
library(caret)
tree = train(Species ~ ., data = training, method = "rf", prox = T)

# boosting
library(ISLR); library(ggplot2); library(caret)
data(Wage)
summary(Wage)

inTrain = createDataPartition(y=Wage$wage, p=0.75, list = F)
training = Wage[inTrain,]
modFit = train(wage ~ ., method="gbm", data=training, verbose=F)

# quiz 3
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
library(dplyr)
# 1
training = segmentationOriginal %>% filter(Case=="Train") %>% select(!Case)
testing = segmentationOriginal %>% filter(Case=="Test") %>% select(!Case)
set.seed(125)
fit = train(Class ~ ., data = training, method = "rpart")
library(rattle)
fancyRpartPlot(fit$finalModel, type = 5, ni=T)

#2
#3
library(pgmm)
data(olive)
olive = olive[,-1]
fit = train(Area ~ ., data = olive, method = "rpart")

newdata = as.data.frame(t(colMeans(olive)))
predict(fit, newdata)

#4
library(ElemStatLearn)
SAheart = read.csv("SAheart.data")
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
set.seed(13234)
library(caret)
fit = train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, data=trainSA, method = "glm", family = "binomial")
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
missClass(testSA$chd, predict(fit, testSA))
missClass(trainSA$chd, predict(fit, trainSA))

#5
vowel.train = read.csv("vowel.train")
vowel.test = read.csv("vowel.test")
vowel.test$y = as.factor(vowel.test$y)
vowel.train$y = as.factor(vowel.train$y)
set.seed(33833)
fit = train(y ~ ., data=vowel.train, method="rf", prox=T)
fit2 = randomForest::randomForest(y ~ ., data=vowel.train, importance=T, proximity=T)

# quiz 4
# 1
vowel.train = read.csv("vowel.train")
vowel.test = read.csv("vowel.test")
vowel.test$y = as.factor(vowel.test$y)
vowel.train$y = as.factor(vowel.train$y)
set.seed(33833)
library(caret)
fitRF = train(y ~ ., data=vowel.train, method="rf", prox=T)
fitGBM = train(y ~ ., data=vowel.train, method="gbm", verbose=F)

predRF = predict(fitRF, vowel.test)
predGBM = predict(fitGBM, vowel.test)

sum(predRF == vowel.test$y)/length(vowel.test$y)
sum(predGBM == vowel.test$y)/length(vowel.test$y)

agreement = which(predGBM == predRF)
sum(predRF[agreement] == vowel.test$y[agreement])/length(agreement)

#2
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
set.seed(62433)

library(caret)
fitRF = train(diagnosis ~ ., data=training, method="rf")
fitGBM = train(diagnosis ~ ., data=training, method="gbm", verbose=F)
fitLDA = train(diagnosis ~ ., data = training, method="lda")

predRF = predict(fitRF, testing)
predGBM = predict(fitGBM, testing)
predLDA = predict(fitLDA, testing)

trainingStacked = data.frame(predRF, predGBM, predLDA, diagnosis = testing$diagnosis)

fitStacked = train(diagnosis ~ ., data=trainingStacked, method="rf")
predStacked = predict(fitStacked, trainingStacked)

sum(predStacked == testing$diagnosis)/length(testing$diagnosis)
sum(predRF == testing$diagnosis)/length(testing$diagnosis)
sum(predGBM == testing$diagnosis)/length(testing$diagnosis)
sum(predLDA == testing$diagnosis)/length(testing$diagnosis)

confusionMatrix(predStacked, testing$diagnosis)

#3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(233)

fitLass = train(CompressiveStrength ~ ., method = "lasso", data=training)
library(elasticnet)
plot.enet(fitLass$finalModel, xvar="penalty")

#4
library(lubridate) # For year() function below
dat = read.csv("gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
library(forecast)
fit = bats(tstrain)
fit %>% forecast(h=235) %>% plot()
fcast = forecast(tstrain, model = fit, h=235)
upperLevel95 = fcast[["upper"]][,2]
sum(testing$visitsTumblr < upperLevel95)/length(upperLevel95)

#5
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(325)
library(e1071)
machine = svm(CompressiveStrength ~ ., data = training)

predMachine = predict(machine, testing)
sqrt(sum((testing$CompressiveStrength - predMachine)^2)/length(predMachine))
