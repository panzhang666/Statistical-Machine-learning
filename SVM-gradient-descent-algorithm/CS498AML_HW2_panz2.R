##Pan Zhang (panz2)
##CS498 HW2

setwd("/Users/panzhang/Desktop/CS498AML/HW2/")
orig_data=read.csv("train.data",header = FALSE)
orig_test_data=read.csv("test.data",header = FALSE)
positive=unique(orig_data[,15])[1]
positive
negative=unique(orig_data[,15])[2]
negative
orig_data=orig_data[,c(1,3,5,11,12,13,15)]
index=sample(2, nrow(orig_data), replace=T, prob=c(0.9,0.1))
train_X = orig_data[index==1, ][,1:(ncol(orig_data)-1)]
train_Y = orig_data[index==1, ][,ncol(orig_data)]
dim(train_X)
val_X = orig_data[index==2, ][,1:(ncol(orig_data)-1)]
val_Y = orig_data[index==2, ][,ncol(orig_data)]
dim(val_X)
test_X=orig_test_data[,c(1,3,5,11,12,13)]
dim(test_X)

#scale variables
for (i in 1:ncol(train_X)){
  train_X[,i]=scale(train_X[,i])
  test_X[,i]=scale(test_X[,i])
  val_X[,i]=scale(val_X[,i])
}
                   

epoch=50
step=360
lambda=c(1e-3, 1e-2, 1e-1, 1)

change_y=function(y){
  if (y==positive){
    return(1)
  }
  else if(y==negative){
    return(-1)
  }
}

f_pred=function(X,a,b){
  return (a %*% t(X) + b)
}

f_update=function(a,b,X,Y,steplen,lambda){
  pred=f_pred(X,a,b) 
  Y=change_y(Y)
  if (pred*Y >= 1){
    a = a - steplen*lambda*a
    b = b
  }
  else{
    a = a - steplen*(lambda*a-Y*X)
    b = b + steplen*Y
  }
  return(list(a,b))
}

magnitude=function(a){
  return(sqrt(sum(a^2)))
}

accuracy=function(X,Y,a,b){
  correct=0
  for (i in 1:nrow(X)){
    pred = f_pred(X[i,],a,b)
    real = change_y(Y[i])
    if (pred*real > 0){
      correct = correct+1
    }
  }
  return (correct/nrow(X))
}
add_quo=function(x){
  return("x")
}

ptm <- proc.time()
step30_acc = array(NA, dim=c(length(lambda),epoch*step/30))
step30_magn = array(NA, dim=c(length(lambda),epoch*step/30))
for (i in 1:length(lambda)) {
  a = c(0,0,0,0,0,0)
  b = 0
  l=lambda[i]
  j=1
  for (e in 1:epoch){
    ev_s=sample(1:nrow(train_X),50,replace = FALSE)
    sample_X= train_X[ev_s,]
    sample_Y=train_Y[ev_s]
    train_X=train_X[-ev_s,]
    train_Y=train_Y[-ev_s]
    step_len=1/(0.01*e+50)
    for (s in 1:step){
      sample_k = sample(1:nrow(train_X),1)
      x_k=train_X[sample_k,]
      y_k=train_Y[sample_k]
      ab_update=f_update(a, b, x_k, y_k, step_len, l)
      a=unlist(ab_update[1])
      b=unlist(ab_update[2])
      if ( s %% 30 == 0){
        step30_acc[i,j]=accuracy(sample_X,sample_Y,a,b) 
        step30_magn[i,j]=magnitude(a)
        j = j+1
      }
    }
  }
  val_accu <- accuracy (val_X, val_Y, a, b)
  print(list(val_accu,l))
  pred_test=array(NA, dim=c(nrow(test_X),1))
  for (t in 1:nrow(test_X)){
    pred_test[t] = f_pred(test_X[t,],a,b)
    if (pred_test[t]>0){
      pred_test[t]="<=50K"
    }
    else{
      pred_test[t]=">50K"
    }
  }
  id= 0:(nrow(test_X)-1)
  id=data.frame(id)
  id=apply(id,1,function(x) paste("\'",x,"\'",sep = ""))
  output = data.frame(id,pred_test)
  names(output) = c("Example","Label")
  csv_name= paste("test_lambda_",l,".csv",sep = "")
  write.table(output, file = csv_name, row.names=F, sep=",")  
} 
proc.time() - ptm

#plot accuracy 
par(mfrow=c(1,1))
plot_acc=step30_acc[1,]
plot(1:length(plot_acc),plot_acc,ylim=c(0,1),type="l", col=1, xlab ="Steps", ylab ="Accuracy", main="Accuracy for different the regularization constants",pch=20)
plot_acc=step30_acc[2,]
lines(1:length(plot_acc),plot_acc, col=2 )
plot_acc=step30_acc[3,]
lines(1:length(plot_acc),plot_acc, col=3 )
plot_acc=step30_acc[4,]
lines(1:length(plot_acc),plot_acc, col=4 )
legend(510,0.25, c("1e-3", "1e-2", "1e-1", "1"), lty = 1, col = c(1,2,3,4),seg.len=0.8,bty="n")

#plot magnitude of the coefficient vector
plot_magn=step30_magn[1,]
plot(1:length(plot_magn) , plot_magn, type="l", col=1, xlab ="Steps", ylab ="Magnitude", main="Magnitude of the coefficient vector  ")
plot_magn=step30_magn[2,]
lines(1:length(plot_magn),plot_magn, col=2 )
plot_magn=step30_magn[3,]
lines(1:length(plot_magn),plot_magn, col=3 )
plot_magn=step30_magn[4,]
lines(1:length(plot_magn),plot_magn, col=4 )
legend("topleft", c("1e-3", "1e-2", "1e-1", "1"), lty = 1, col = c(1,2,3,4),seg.len=0.8,bty="n")

