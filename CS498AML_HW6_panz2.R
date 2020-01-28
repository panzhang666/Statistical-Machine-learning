library(MASS)
setwd('/Users/panzhang/Desktop/CS498AML/HW6')
data=read.table("housing.data",header = FALSE)
measure=data.matrix(data[,1:13])
price=data[,14]
model=lm(price ~ measure)
#plot(model,which=c(1:6))
stdres = rstandard(model)
plot(model$fitted.values, stdres, 
        ylab="Standardized Residuals", 
        xlab="Fitted values", 
        main="Without any transforms")
plot(price,model$fitted.values,
     ylab="Predicted house price",
     xlab="True house price")
abline(0,1,col="red")

#Remove outlier points
data1=data[-c(369,373,372),]
measure1=data.matrix(data1[,1:13])
price1=data1[,14]
model1=lm(price1 ~ measure1)
#plot(model1,which=c(1:6))
data2=data1[-c(366,369),]
measure2=data.matrix(data2[,1:13])
price2=data2[,14]
model2=lm(price2 ~ measure2)
#plot(model2,which=c(1:6))
data3=data2[-c(367,368,408),]
measure3=data.matrix(data3[,1:13])
price3=data.matrix(data3[,14])
model3=lm(price3 ~ measure3)
#plot(model3,which=c(1:6))

#Box-Cox transformation
bc = boxcox(price3 ~ measure3)
I=which(bc$y==max(bc$y))
lambda=bc$x[I]
lambda
new_model=lm(price3^lambda~measure3)
#plot(new_model)
res3=new_model$residuals
stdres3 = rstandard(new_model)
plot(new_model$fitted.values, stdres3, 
     ylab="Standardized Residuals", 
     xlab="Fitted values", 
     main="After removing and transforming")
plot(new_model$fitted.values^(1/lambda), stdres3, 
     ylab="Standardized Residuals", 
     xlab="Fitted values (new_model$fitted.values^(1/lambda))", 
     main="After removing and transforming")
plot(price3,new_model$fitted.values^(1/lambda),
     ylab="Predicted house price",
     xlab="True house price")
abline(0,1,col="red") 

