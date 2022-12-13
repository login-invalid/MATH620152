setwd("/Users/vielyi/Desktop/课程/统计软件/project/2022期末数据")
rm(list = ls()) # 清除变量
data = read.csv("departements_cas.csv",header = T)

#######################
# 主成分分析 #########
#######################

data_sub = data[,c(3:6)]
data1 = scale(data_sub)
data.pr = princomp(data1 ,cor = FALSE, scores = TRUE)
summary(data.pr,loadings=TRUE)

data.pr1 = prcomp(data1,center = TRUE, scale. = TRUE) # 获取主成分
data.pr1
summary(data.pr1)

screeplot(data.pr,type = "lines") # 主成分（4个主成分权重图）
pre = predict(data.pr)
pre

#######################
# 支持向量机 ##########
#######################
library(e1071)
str(data) # 查看dataframe属性类型
data$KÉDOUGOU = as.factor(data$KÉDOUGOU)
fit = svm(formula = KÉDOUGOU ~ MBACKÉ + GUÉDIAWAYE,
           data = data, 
           type = "C-classification", #分类
           kernel = "sigmoid", #选择核函数 
           scale = TRUE,
           gamma = 1,
           cost = 0.2)
summary(fit)

## 自动调参
tuning = tune(svm,
              KÉDOUGOU ~ MBACKÉ + GUÉDIAWAYE,
              data = data, 
              kernel = "sigmoid", 
              type = "C-classification",
              ranges = list(
               cost = c(0.001, 0.01, 0.1, 1), 
               gamma = seq(1:5)),
              tunecontrol = tune.control(
               sampling = "cross", # 选择标准，交叉验证k-folded
               cross = 20),
              probability = TRUE)
summary(tuning)
best = tuning$best.model # 获取调参得到的最优模型
tuning$best.parameters
summary(best)

predict(best, data.frame(MBACKÉ = c(300,400,500),GUÉDIAWAYE = c(100,150,200)))
predict(best, data.frame(MBACKÉ = c(300,400,500),
                         GUÉDIAWAYE = c(100,150,200)),probability = TRUE)


#######################
# 神经网络 ############
#######################
library(neuralnet)

rm(list = ls())
data = read.csv("NMDS_coordinates.csv",header = T)
str(data)

data = data[,c(2,30:33)]
str(data)
normalize = function(x){
  return((x-min(x))/(max(x)-min(x)))
}
data_norm = as.data.frame(lapply(data,normalize)) #lapply()将函数运用于每一列
summary(data_norm)

## 随机抽样分数据集为训练集和测试集
set.seed(1)
index = sample(nrow(data_norm),nrow(data_norm)*.7,replace=F)
train_set = data_norm[index,]
test_set = data_norm[-index,]

## 训练模型
model = neuralnet(NMDS1 ~ days + Vel + X_Vel + MidPt, data = train_set)

model_results = compute(model,test_set[1:4])
predicted_NMDS1 = model_results$net.result
cor(predicted_NMDS1,test_set$NMDS1)

plot(model)

ntest = length(test_set$NMDS1)
ntrain = length(train_set$NMDS1)
train_sse = model$result.matrix[1]
predict_sse = sum(1/2*(predicted_NMDS1-test_set$NMDS1)^2)
c(variance = predict_sse/(ntest-5),bias = train_sse/(ntrain-5))

plot(model,radius=0.2,arrow.length = 0.15,
     col.entry.synapse="violet",
     col.entry="purple",
     col.hidden="grey",
     col.hidden.synapse="grey",
     col.out="cyan4",
     col.out.synapse="cyan4",
     col.intercept="cyan",
     fontsize=10)

## 更多隐藏层和更改激活函数
model2 = neuralnet(NMDS1 ~ days + Vel + X_Vel + MidPt, data = train_set,
                   hidden=c(5,5),
                   act.fct =  "logistic")
model_results = compute(model2,test_set[1:4])
predicted_NMDS1 = model_results$net.result
cor(predicted_NMDS1,test_set$NMDS1)
plot(model2)


#######################
# 决策树 ##############
#######################
library(rpart)
library(rpart.plot)

rm(list = ls())
data = read.csv("NMDS_coordinates.csv",header = T)
str(data)
data$SEASON = as.factor(data$SEASON)
data$SITE = as.factor(data$SITE)

## 分类树
formula = rpart(formula = NMDS2 ~ SEASON + SITE + PteCal, data = data,
                weights = abs(rnorm(nrow(data))), na.action = na.pass)
rpart.plot(formula, fallen.leaves = F, main = "fallen.leaves = F")
formula$frame

## 回归树
## 以 gini 指标为标准
formula2 = rpart(formula = NMDS2 ~ SEASON + SITE + PteCal, data = data,
                 method = "anova",parms = list(split = "gini")) 
rpart.plot(formula2, shadow.col = "gray", main = "shadow.col = gray")
formula2$frame
summary(formula2)


#######################
# GLM #################
#######################
library(arules)
library(pROC)
library(MASS)

rm(list = ls())
data = read.csv("confirmes.csv",header = T)
summary(data)

Poisson = glm(evacue ~ gueri + mort + communautaire, data = data, family = poisson())
summary(Poisson)

Poisson2 = glm(evacue ~ gueri + mort, data = data, family = quasipoisson())
summary(Poisson2) # AIC: NA ?

## 模型选择
anova(Poisson , Poisson2, test = "Chisq")
stepAIC(Poisson)

## ROC曲线
pre = predict(Poisson, type = "response")
modeltest =  roc(data$evacue, pre)
plot(modeltest, print.auc = TRUE, max.auc.polygon = TRUE)

#######################
# 梯度提升树 ##########
#######################
library(gbm)
library(caret)

rm(list = ls())
data = read.csv("NMDS_coordinates.csv",header = T)
data = data[,-c(3:12)]
data = data[, - 21]

## 检视数据并分割数据集
str(data)
dim(data)
head(data)

train = createDataPartition(data$DIST, p = 0.3, list=FALSE)
train_set = data[train,]
test_set = data[-train,]

## GBRT梯度提升树

model = gbm(DIST~.,
            distribution = 'gaussian',
            data = data,
            n.trees=1000,
            shrinkage = 0.001)

model.best.iter = gbm.perf(model,
                           plot.it = TRUE,
                           oobag.curve = TRUE,
                           method = "OOB")

summary.gbm(model)

model.test = predict(model,
                    newdata = test_set,
                    n.trees = model.best.iter)
plot(test_set$DIST, model.test, main = '',
                xlab = 'Original', ylab = 'Predict')
abline(1, 1)

#######################
# 基本作图 ############
#######################
rm(list = ls())

data = read.csv("regions_cas.csv",header = T)
str(data)

## 人为增加两列列分类数据
data$category =sample(factor(c("A", "B", "C")),length(data$MATAM),replace = T)
data$category2 =sample(factor(c("D1", "D2", "D3")),length(data$MATAM),replace = T)

## 柱状图 
ret = table(data$category)
barplot(ret)
barplot(ret, 
        width=0.5, xlim=c(-3, 5),
        main="NAME1", 
        col=c("brown2", "aquamarine1","orange2"))

ret = table(data$FATICK,data$category)
barplot(ret,
        names.arg = c("T1","T2","T3"), 
        col = colors()) #调整颜色和标签

ret = table(data$category,data$category2)
barplot(ret,
        col = c('orange', 'steelblue', 'red'),
        border = c('#fff5ee', '#00ff7f', '#6a5acd'), 
        legend=TRUE) #列联表
barplot(ret, beside=TRUE, legend=TRUE,
        ylim=c(0, 25),
        xlim=c(-1, 10), 
        width=0.6,
        col= c('orange', 'steelblue', 'red')) #分列

hist(data$MATAM)
hist(data$MATAM, col=rainbow(15), 
     main='NAME2', xlab='XLABEL', ylab='YLABEL')

dens =  density(data$MATAM)
hist(data$MATAM, freq=FALSE,
     ylim=c(0,max(dens$y)),
     col=rainbow(15),
     main='NAME2',
     xlab='XLABEL', ylab='YLABEL')
lines(dens, lwd=2, col='blue') # 添加核密度曲线

par(mfrow = c(1, 3))
hist(data$MATAM,  density = 2, angle = 45,  main  = "angle = 45")
hist(data$MATAM,  density = 2, angle = 90,  main  = "angle = 90")
hist(data$MATAM,  density = 2, angle = 180, main  = "angle = 180")

## 箱形图
par(mfrow = c(1, 2))
boxplot(data$MATAM ~ data$category)
boxplot(data$MATAM ~ data$category2, varwidth = TRUE, names = c("N1","N2","N3"))

## 散点图
par(mfrow = c(1, 1))
plot(data$DAKAR,data$MATAM)
plot(data$DAKAR, data$MATAM,
     pch=2, col='blue',
     cex=1)
plot(data$DAKAR, data$MATAM,
     pch=2, col='blue',
     cex = (data$DAKAR)/(data$MATAM+data$DAKAR))
plot(data$DAKAR, data$MATAM, 
     type = "l", panel.last = grid(5, 10),
     pch = 0, lwd = 4, col = "blue",
     main = "NAME3")

## 饼状图
ret = table(data$category)
labels = c("a", "b", "b")
pie(ret,labels,col = rainbow(3))

#######################
# 偏最小二乘 ##########
#######################
library(pls)
rm(list = ls())
data = read.csv("NMDS_coordinates.csv",header = T)
str(data)

data = data[,c(2,30:33)]
str(data)
normalize = function(x){
  return((x-min(x))/(max(x)-min(x)))
}
data_norm = as.data.frame(lapply(data,normalize))
summary(data_norm)
## 随机抽样分数据集为训练集和测试集
set.seed(1)
index = sample(nrow(data_norm),nrow(data_norm)*.7,replace=F)
train_set = data_norm[index,]
test_set = data_norm[-index,]

##先进行线性回归
attach(train_set)
lr_model = lm(NMDS1 ~ days+Vel+X_Vel+MidPt)
pred_lr = predict(lr_model, newdata = test_set)
RMSE_lr = sqrt(mean((test_set$NMDS1-pred_lr)^2))
RMSE_lr
detach(train_set)

## 然后进行主成分回归
attach(train_set)
pcr_mode1 = pcr(NMDS1 ~ days+Vel+X_Vel+MidPt,scale=TRUE,validation="CV")
validationplot(pcr_mode1,val.type="RMSEP")
summary(pcr_mode1)

## 选用2个进行主成分
pcr_model2 = pcr(NMDS1 ~ days+Vel+X_Vel+MidPt,scale=TRUE,validation="CV",ncomp=2)
summary(pcr_model2)
coef(pcr_model2)
pred_pcr2 = predict(pcr_model2,newdata = test_set,ncomp = 2)
RMSE_pcr2 = sqrt(mean((test_set$NMDS1-pred_pcr2)^2))
RMSE_pcr2
detach(train_set)

## pls偏最小二乘
attach(train_set)
pls_model = plsr(NMDS1 ~ days+Vel+X_Vel+MidPt,validation="LOO",scale=T,jackknife=T) 
summary(pls_model)
plot(RMSEP(pls_model))

## 2个成分
pls_model2 = plsr(NMDS1 ~ days+Vel+X_Vel+MidPt,ncomp=2,scale=T,validation="LOO")
summary(pls_model2)
loading.weights(pls_model2)
coef(pls_model2)
pred_plsr2 =predict(pls_model2,newdata = test_set,ncomp=2)
RMSE_plsr2 = sqrt(mean((test_set$NMDS1-pred_plsr2)^2))
RMSE_plsr2

## 3个成分
pls_model3 = plsr(NMDS1 ~ days+Vel+X_Vel+MidPt,ncomp=3,scale=T,validation="LOO")
summary(pls_model3)
loading.weights(pls_model3)
coef(pls_model3)
pred_plsr3 = predict(pls_model3,newdata = test_set,ncomp = 3)
RMSE_plsr3 = sqrt(mean((test_set$NMDS1-pred_plsr3)^2))
RMSE_plsr3
detach(train_set)
