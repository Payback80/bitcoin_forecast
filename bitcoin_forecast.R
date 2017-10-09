library(Quandl)
library(TTR)
library(quantmod)
library(caret)
library(nnet)


# retrive data from Quandl 
data = Quandl("BCHARTS/BITSTAMPUSD")
data2 <- data[order(as.Date(data$Date, format = "%Y-%m-%d")), ]
price <- data2$Close

HLC<-matrix(c(data2$High, data2$Low, data2$Close),nrow=length(data2$High))

bitcoin.lr <- diff(log(price))

#creating indicators 
rsi <- RSI(price)
MACD <- MACD(price)
macd <- MACD[, 1]
will <- williamsAD(HLC)
cci <- CCI(HLC)
STOCH <- stoch(HLC)
stochK <- STOCH[, 1]
stochD <- STOCH[, 1]

#create input and target matrix
Input<-matrix(c(rsi[400:939], cci[400:939], macd[400:939], will[400:939], stochK[400:939], stochD[400:939]),nrow=540)
Target<-matrix(c(bitcoin.lr[401:940]), nrow=540)



trainingdata <- cbind(Input,Target)
colnames(trainingdata) <- c("RSI","CCI","MACD","WILL","STOCHK","STOCHD", "Return")



# split the dataset 90-10% ratio
trainIndex <- createDataPartition(bitcoin.lr[401:940], p=.9, list=F)
bitcoin.train <- trainingdata[trainIndex, ]
bitcoin.test <- trainingdata[-trainIndex, ]


# derive the best neural network model using rmse criteria, i neuron hidden layer j learning rate
best.network<-matrix(c(5,0.5))
best.rmse<-1
for (i in 5:15) for (j in 1:5) {
  bitcoin.fit <- nnet(Return ~ RSI + CCI + MACD + WILL + STOCHK + STOCHD, data = bitcoin.train, 
                      maxit=2000, size=i, decay=0.001*j, linout = 1)    # alpha learning rate little step gradient descent
  
  bitcoin.predict <- predict(bitcoin.fit, newdata = bitcoin.test)
  bitcoin.rmse <- sqrt(mean((bitcoin.predict - bitcoin.lr[917:940])^2)) 
  if (bitcoin.rmse<best.rmse) {
    best.network[1,1]<-i
    best.network[2,1]<-j
    best.rmse<-bitcoin.rmse  
  }
}

# create the Input and Target matrix for test
InputTest<-matrix(c(rsi[1710:1799], cci[1710:1789], macd[1710:1799], will[1710:1799], stochK[1710:1799], stochD[1710:1799]),nrow=90)
TargetTest<-matrix(c(bitcoin.lr[1711:1800]), nrow=90)



Testdata <- cbind(InputTest,TargetTest)
colnames(Testdata) <- c("RSI","CCI","MACD","WILL","STOCHK","STOCHD", "Return")



# fit the best model on test data
bitcoin.fit <- nnet(Return ~ RSI + CCI + MACD + WILL + STOCHK + STOCHD, data = trainingdata, 
                    maxit=2000, size=best.network[1,1], decay=0.1*best.network[2,1], linout = 1) 

bitcoin.predict1 <- predict(bitcoin.fit, newdata = Testdata)

# repeat and average the model 200 times  
for (i in 1:20) {
  bitcoin.fit <- nnet(Return ~ RSI + CCI + MACD + WILL + STOCHK + STOCHD, data = trainingdata, 
                      maxit=1000, size=best.network[1,1], decay=0.1*best.network[2,1], linout = 1) 
  
  bitcoin.predict<- predict(bitcoin.fit, newdata = Testdata)
  bitcoin.predict1<-(bitcoin.predict1+bitcoin.predict)/2
}


# calculate the buy-and-hold benchmark strategy and neural network profit on the test dataset
money<-matrix(0,90)
money2<-matrix(0,90)
money[1,1]<-100
money2[1,1]<-100
for (i in 2:90) {
  if (bitcoin.predict1[i-1]<0) {
    direction1<--1  
  } else {
    direction1<-1}
  if (TargetTest[i-1]<0) {
    direction2<--1  
  } else {
    direction2<-1 }
  if ((direction1-direction2)==0) {
    money[i,1]<-money[i-1,1]*(1+abs(TargetTest[i-1]))  
  } else {
    money[i,1]<-money[i-1,1]*(1-abs(TargetTest[i-1])) }
  money2[i,1]<-100*(price[1710+i-1]/price[1710])
}


#plot benchmark and neural network profit on the test dataset
x<-1:91
matplot(cbind(money, money2), type = "l", xaxt = "n", ylab = "")
legend("topright", legend = c("Neural network","Benchmark"), pch = 19, col = c("black", "red"))



