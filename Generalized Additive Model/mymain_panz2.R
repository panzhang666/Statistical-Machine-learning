if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  "glmnet",
  "xgboost",
  "mgcv"
)

setwd("/Users/panzhang/Desktop/STAT542/project1_JL")
rawdata<-read.csv('Ames_data.csv')
categorical.vars <- colnames(rawdata)[which(sapply(rawdata,function(x) is.factor(x)))]
nmatrix <- rawdata[, !colnames(rawdata) %in% categorical.vars, drop=FALSE]
summary(nmatrix)
for(var in categorical.vars){
  mylevels <- sort(unique(rawdata[, var]))
  m <- length(mylevels)
  print(c(var,m))
}

ProcessMissingFixVal <-function(train.column,test.column,replced_n){
  idtrain = which(is.na(train.column))
  idtest=which(is.na(test.column))
  train.column[idtrain]=replced_n
  test.column[idtest]=replced_n
  return(list(train = train.column, test = test.column))
}

PreProcessingMatrixOutput <- function(train.data, test.data){
  # generate numerical matrix of the train/test
  # assume train.data, test.data have the same columns
  categorical.vars <- colnames(train.data)[which(sapply(train.data,function(x) is.factor(x)))]
  train.matrix <- train.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  test.matrix <- test.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  n.train <- nrow(train.data)
  n.test <- nrow(test.data)
  for(var in categorical.vars){
    mylevels <- sort(unique(train.data[, var]))
    m <- length(mylevels)
    tmp.train <- matrix(0, n.train, m)
    tmp.test <- matrix(0, n.test, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[train.data[, var]==mylevels[j], j] <- 1
      tmp.test[as.character(test.data[, var])==as.character(mylevels[j]), j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    colnames(tmp.test) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
    test.matrix <- cbind(test.matrix, tmp.test)
  }
  return(list(train = train.matrix, test = test.matrix))
}

ProcessingExtremeValue <- function(trainY,testY,thred=0.05){
  lim_train <- quantile(trainY, probs=c(thred, 1-thred))
  lim_test <- quantile(testY, probs=c(thred, 1-thred))
  trainY[trainY < lim_train[1]]=lim_train[1]
  trainY[trainY > lim_train[2]]=lim_train[2]
  testY[testY < lim_test[1]]=lim_test[1]
  testY[testY > lim_test[2]]=lim_test[2]
  return(list(train = trainY, test = testY))
}

data <- read.csv("Ames_data.csv")
load("project1_testIDs.R")
j=8
train <- data[-testIDs[,j], ]
test <- data[testIDs[,j], ]
remove.var <- c('PID', 'Sale_Price')
train.y <- log(train$Sale_Price)
train.PID<-train$PID
train.x <-train[, !colnames(train) %in% remove.var, drop=FALSE]
test.y <- log(test$Sale_Price)
test.PID <- test$PID
test.x <- test[, !colnames(test) %in% remove.var, drop=FALSE]
r <- ProcessMissingFixVal(train.x$Garage_Yr_Blt, test.x$Garage_Yr_Blt, 0)
train.x$Garage_Yr_Blt <- r$train
test.x$Garage_Yr_Blt <- r$test
r <- ProcessingExtremeValue(train.y, test.y, 0.05)
train.y <-r$train
test.y <- r$test

pre.matrix <- PreProcessingMatrixOutput(train.x, test.x)
train.x.pre=pre.matrix$train
test.x.pre=pre.matrix$test
train_x=as.matrix(train.x.pre)
test_x=as.matrix(test.x.pre)

library(glmnet)
## linear regression models with Lasso 

mylasso = glmnet(train_x, train.y, alpha = 1)
cv.out = cv.glmnet(test_x, test.y, alpha = 1)
pred = predict(mylasso, s=cv.out$lambda.min,newx=test_x)
sqrt(mean((pred - test.y)^2))

## boosting tree method 

library(xgboost)
rfModel = xgboost(data = train_x, label = train.y , eta = 0.2, nrounds = 30)
tmp <- predict(rfModel, test_x)
sqrt(mean((tmp - test.y)^2))

## GAM

library(mgcv)
remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating',
                'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
                'Longitude','Latitude', 'Mo_Sold', 'Year_Sold',
                'PID', 'Sale_Price')
linear.vars <- c('BsmtFin_SF_1', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 
                 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 
                 'Kitchen_AbvGr', 'Fireplaces', 'Garage_Cars')

categorical.vars <- colnames(train.x)[which(sapply(train.x, function(x) is.factor(x)))]
num.vars <- names(train.x)
num.vars <- num.vars[num.vars != "Sale_Price"]
num.vars <- num.vars[! num.vars %in% categorical.vars]
num.vars <- num.vars[! num.vars %in% linear.vars]

select.level.var = c('MS_SubClass__Duplex_All_Styles_and_Ages', 
                     'MS_SubClass__One_Story_1945_and_Older',
                     'MS_SubClass__Two_Story_PUD_1946_and_Newer',
                     'MS_Zoning__C_all', 'MS_Zoning__Residential_Medium_Density',
                     'Neighborhood__Crawford', 'Neighborhood__Edwards',
                     'Neighborhood__Green_Hills', 'Neighborhood__Meadow_Village',
                     'Neighborhood__Northridge', 'Neighborhood__Somerset', 
                     'Neighborhood__Stone_Brook','Overall_Qual__Above_Average',
                     'Overall_Qual__Average','Overall_Qual__Good',
                     'Overall_Qual__Very_Good','Overall_Qual__Excellent',
                     'Overall_Qual__Below_Average', 'Overall_Qual__Fair',
                     'Overall_Qual__Poor','Overall_Qual__Very_Excellent',
                     'Overall_Qual__Very_Poor','Overall_Cond__Average',
                     'Overall_Cond__Above_Average', 'Overall_Cond__Good', 
                     'Overall_Cond__Poor','Overall_Cond__Very_Good',
                     'Overall_Cond__Below_Average','Overall_Cond__Fair',
                     'Overall_Cond__Very_Poor', 'Overall_Cond__Excellent')
m <- length(select.level.var)
tmp.train <- matrix(0, nrow(train.x), m)
tmp.test <- matrix(0, nrow(test.x), m)
colnames(tmp.train) <- select.level.var
colnames(tmp.test) <- select.level.var
for(i in 1:m){
  tmp <- unlist(strsplit(select.level.var[i], '__'))
  select.var <- tmp[1]
  select.level <- tmp[2]
  tmp.train[train.x[, select.var]==select.level, i] <- 1
  tmp.test[test.x[, select.var]==select.level, i] <- 1
}

gam.formula <- paste0("Sale_Price ~ ",linear.vars[1])

for(var in c(linear.vars[-1], colnames(tmp.train))){
  gam.formula <- paste0(gam.formula, " + ", var)
}
for(var in num.vars){
  gam.formula <- paste0(gam.formula, " + s(", var, ", k=5)")
}
gam.formula <- as.formula(gam.formula)

Sale_Price=train.y
train.new=cbind(train.x,tmp.train,Sale_Price)
attach(train.new)
gam.model <- gam(gam.formula, data = train.new, method="REML")

Sale_Price=test.y
test.new=cbind(test.x,tmp.test,Sale_Price)
attach(test.new)
tmp <- predict.gam(gam.model, newdata = test.new)
sqrt(mean((tmp - test.y)^2))
