Estep = function(data, G, para){
  pr = para$prob
  mu = para$mean
  Sinv = solve(para$Sigma)
  n = nrow(data)
  tmp = NULL
  for(k in 1:G){
    tmp = cbind(tmp, 
                apply(data, 1, 
                      function(x) t(x - mu[, k]) %*% Sinv %*% (x - mu[, k])))
  }
  tmp = -tmp/2 + matrix(log(pr), nrow=n, ncol=G, byrow=TRUE)
  # tmp = tmp - apply(tmp, 1, mean)
  # bigM = 15
  # tmp[tmp > bigM] = bigM
  # tmp[tmp < -bigM] = -bigM
  tmp = exp(tmp)
  tmp = tmp / apply(tmp, 1, sum)
  return(tmp)
}


Mstep <- function (data, G, para, post.prob ) {
  # Your Code
  # Return the updated parameters
  n = nrow(data)
  m = ncol(data)
  update.prob = apply(post.prob, 2, sum)/n
  update.mu = NULL
  update.sigma = array(0,dim=c(2,2))
  num.sigma = array(0,dim=c(2,2))
  denom.sigma = 0
  for (k in 1:G){
    num.mu = rep(0,m)
    denom = 0
    for (i in 1:n){
      num.mu = num.mu + post.prob[i,k]*data[i,]
      denom = denom + post.prob[i,k]
    }
    new.mu = num.mu/denom
    #print(new.mu)
    update.mu = cbind(update.mu,t(new.mu))
    for (i in 1:n){
      num.sigma = num.sigma + post.prob[i,k]* as.numeric(data[i,] - new.mu) %*% t(as.numeric(data[i,] - new.mu))
    }
    denom.sigma = denom.sigma + denom
  }
  update.sigma = num.sigma/denom.sigma
  return(list(prob=update.prob, mean=update.mu, Sigma= update.sigma))
}

myEM <- function (data, T, G, para ) {
  for(t in 1: T ) {
    post.prob <- Estep(data, G, para )
    para <- Mstep (data, G, para, post.prob )
  }
  return (para)
}


library(mclust)
n <- nrow(faithful)
Z <- matrix (0, n, 2)
Z[sample (1:n, 120), 1] <- 1
Z[, 2] <- 1 - Z[, 1]
ini0 <- mstep(modelName ="EEE", faithful, Z)$parameters
# Output from my EM alg
para0 <- list(prob = ini0$pro, mean=ini0$mean, Sigma = ini0$variance$Sigma)
myEM (data = faithful, G = 2, T = 10, para = para0 )
# Output from mclust
Rout <- em (modelName = "EEE", data = faithful, control = emControl(eps =0, tol =0, itmax = 10), parameters = ini0 )$parameters
list ( Rout$pro, Rout$mean, Rout$variance$Sigma )