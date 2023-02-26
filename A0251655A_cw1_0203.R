library(readxl)
library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)
library(lubridate)
library(ggpubr)
library(epitools)
library(MASS)
library(caret)
library(pROC)
library(ranger)
library(gbm)
library(kernlab)
library(themis)
getwd()
setwd("~/Downloads/SPH6004/assignment1")
set.seed(20230217)
#### eda ####
data1 <- read.csv("Assignment_1_data.csv")
data1 <- as.data.table(data1)
dim(data1)
summary(data1)
unique(data1$outcome)
colnames(data1)
data1[,outcome1:=as.factor(outcome)]
summary(data1$outcome1)

##### missing data removal #####
column_na <- data.table(column_index=1:dim(data1)[2],column_name=colnames(data1))
data1 <- as.data.frame(data1)
for (j in 1:dim(column_na)[1]){
  na_perc1=mean(as.integer(is.na(data1[,j])))
  set(column_na,j,"na_perc",na_perc1)
}
summary(column_na$na_perc)

## remove columns with missing percentage larger than 0.45
## Missing Data Estimation in Morphometrics: How Much is Too Much? : simulation tests shown that MI algorithms had relatively bad effect with missing rate larger than 40% 
col_remove <- column_na[which(column_na[,na_perc>=0.45]),]$column_index
data2 <- data1[,-col_remove]

data2 <- as.data.table(data2)
for (i in 1:dim(data2)[1]){
  row1 <- data2[i,]
  na_perc <- mean(as.integer(is.na(row1)))
  set(data2,i,"na_perc",na_perc)
}
summary(data2$na_perc)
dim(data2[which(data2[,na_perc>=0.5])])

## remove rows with missing percentage larger than 0.5
data3 <- data2[which(data2[,na_perc<0.5]),]

##### missing value impute with median values #####
data3

for (k in 2:(dim(data3)[2]-2)){
  median1 <- quantile(data3[,..k],0.5,na.rm=TRUE)
  set(data3,which(is.na(data3[,..k])),colnames(data3)[k],median1)
}
summary(data3)

##### z score normalization #####
data3[,outcome:=factor(outcome)]
data3[,gender:=factor(gender)]
summary(data3)
colnames(data3)

data4 <- data3[,1:34]
colnames(data4)

for (k in 2:(dim(data4)[2]-1)){
  mean1 <- mean(data4[[colnames(data4)[k]]])
  sd1 <- sd(data4[[colnames(data4)[k]]])
  set(data4,NULL,colnames(data4)[k],(data4[,..k]-mean1)/sd1)
}
summary(data4)

saveRDS(data4,"preprocess0215.rds")
#data4 <- readRDS("preprocess0215.rds")

#### feature selection ####
modelnull = glm(outcome~1,data=data4,family = "binomial")
modelfull = glm(outcome~.,data=data4,family = "binomial")
predict(modelfull,newdata = data4[,1:33])
step_forward <- step(object=modelnull,direction = "forward",scope=list(lower=modelnull,upper=modelfull))
# step forward selects 17 variables
step_backward <- step(object=modelfull,direction = "backward",scope=list(lower=modelnull,upper=modelfull))
# step backward selects 20 variables
saveRDS(step_forward,"step_forward0215.rds")
saveRDS(step_backward,"step_backward0215.rds")
#step_forward <- readRDS("step_forward0215.rds")
#step_backward <- readRDS("step_backward0215.rds")

name_forward <- names(step_forward$coefficients)
name_backward <- names(step_backward$coefficients)

name_union <- c(name_forward[!name_forward%in%name_backward],name_backward)
name_union <- name_union[-2]
# the selected variables are 21
name_union[2]="gender"
name_union <- c(name_union,"outcome")

data5 <- data4[,..name_union]
saveRDS(data5,"data_after_feature_selection.rds")

#### predictice model ####
data5 <- readRDS("data_after_feature_selection.rds")
data5 <- as.data.frame(data5)
train_index <- createDataPartition(data5$outcome,p=0.8)
train1 <- data5[train_index$Resample1,]
valid1 <- data5[-train_index$Resample1,]
summary(train1)
summary(valid1)

##### rf #####
# for rf, can use out of bag error, so no need to do 5-fold cross validation
model_selection_rf <- expand.grid(model="random forest",K_smote=3:6,num_of_tree=c(100,200,300,400,500),obb_error=0)

for (model in 1:dim(model_selection_rf)[1]){
  K1 <- model_selection_rf[model,]$K_smote
  ntree <- model_selection_rf[model,]$num_of_tree
  train2 <- smotenc(train1,var="outcome",k=K1)
  rf_model1 <- ranger(outcome~.,data=train2,importance = "impurity",probability = TRUE,num.trees=ntree)
  model_selection_rf[model,]$obb_error <- rf_model1$prediction.error
}

model_selection_rf2 <- expand.grid(model="random forest",K_smote=0,num_of_tree=c(100,200,300,400,500),obb_error=0)

for (model in 1:dim(model_selection_rf2)[1]){
  ntree <- model_selection_rf2[model,]$num_of_tree
  rf_model1 <- ranger(outcome~.,data=train1,importance = "impurity",probability = TRUE,num.trees=ntree)
  model_selection_rf2[model,]$obb_error <- rf_model1$prediction.error
}
model_selection_rf <- rbind(model_selection_rf2,model_selection_rf)
saveRDS(model_selection_rf,"model_selection_rf.rds")


model_selection_rf <- as.data.table(model_selection_rf)
model_selection_rf[,K_smote1:=factor(K_smote,levels = c(0,3,4,5,6),labels = c("no SMOTE","SMOTE(K=3)","SMOTE(K=4)","SMOTE(K=5)","SMOTE(K=6)"))]

p1 <- ggplot(model_selection_rf)+
  geom_point(aes(x=num_of_tree,y=obb_error,color=K_smote1))+
  geom_line(aes(x=num_of_tree,y=obb_error,color=K_smote1))+
  theme_bw()+
  labs(x="number of trees",y="out of bag error",color="choices of K in SMOTE",title = "hyperparameter tuning in random forest")

# selection K=3, ntree=400

ggsave("rf_hyper_selection.pdf",p1,w=8,h=6)

##### gradient boosting #####
model_selection_gb <- expand.grid(model="gradient boost",K_smote=3:6,num_of_tree=c(100,200,300,400,500,600,700,800),cv_error=0)

for (model in 1:dim(model_selection_gb)[1]){
  K1 <- model_selection_gb[model,]$K_smote
  train2 <- smotenc(train1,var="outcome",k=K1)
  train2 <- as.data.table(train2)
  set(train2,NULL,"outcome",train2[,as.integer(outcome)])
  set(train2,NULL,"outcome",train2[,outcome-1])
  
  cv1 <- createFolds(train2$outcome,k=5)
  trainset <- train2[-cv1[[1]],]
  testset <- train2[cv1[[1]],]
  
  n_tree <- model_selection_gb[model,]$num_of_tree
  
  gb_model1 <- gbm(outcome~.,data=trainset,distribution="bernoulli",n.trees = n_tree)
  pre1 <- predict(gb_model1,newdata = testset[,-22],n.trees = n_tree,type="response")
  
  gb_roc1 <- roc(response=testset$outcome,predictor=pre1)
  gb_t1 <- coords(gb_roc1,"best", ret = "threshold")$threshold
  
  error_sum <- mean(as.integer(pre1>=gb_t1)!=testset$outcome)
  
  for (i in 2:5){
    trainset <- train2[-cv1[[i]],]
    testset <- train2[cv1[[i]],]
    
    gb_model1 <- gbm(outcome~.,data=trainset,distribution="bernoulli",n.trees = n_tree)
    pre1 <- predict(gb_model1,newdata = testset[,-22],n.trees = n_tree,type="response")
    
    gb_roc1 <- roc(response=testset$outcome,predictor=pre1)
    gb_t1 <- coords(gb_roc1,"best", ret = "threshold")$threshold
    
    error_sum <- error_sum+mean(as.integer(pre1>=gb_t1)!=testset$outcome)
  }
  
  model_selection_gb[model,]$cv_error <- error_sum/5
}


model_selection_gb2 <- expand.grid(model="gradient boost",K_smote=0,num_of_tree=c(100,200,300,400,500,600,700,800),cv_error=0)

cv2 <- createFolds(train1$outcome,k=5)
for (model in 1:dim(model_selection_gb2)[1]){
  train2 <- as.data.table(train1)
  set(train2,NULL,"outcome",train2[,as.integer(outcome)])
  set(train2,NULL,"outcome",train2[,outcome-1])
  
  trainset <- train2[-cv2[[1]],]
  testset <- train2[cv2[[1]],]
  
  n_tree <- model_selection_gb2[model,]$num_of_tree
  
  gb_model1 <- gbm(outcome~.,data=trainset,distribution="bernoulli",n.trees = n_tree)
  pre1 <- predict(gb_model1,newdata = testset[,-22],n.trees = n_tree,type="response")
  
  gb_roc1 <- roc(response=testset$outcome,predictor=pre1)
  gb_t1 <- coords(gb_roc1,"best", ret = "threshold")$threshold
  
  error_sum <- mean(as.integer(pre1>=gb_t1)!=testset$outcome)
  
  for (i in 2:5){
    trainset <- train2[-cv2[[i]],]
    testset <- train2[cv2[[i]],]
    
    gb_model1 <- gbm(outcome~.,data=trainset,distribution="bernoulli",n.trees = n_tree)
    pre1 <- predict(gb_model1,newdata = testset[,-22],n.trees = n_tree,type="response")
    gb_roc1 <- roc(response=testset$outcome,predictor=pre1)
    gb_t1 <- coords(gb_roc1,"best", ret = "threshold")$threshold
    
    error_sum <- error_sum+mean(as.integer(pre1>=gb_t1)!=testset$outcome)
  }
  
  model_selection_gb2[model,]$cv_error <- error_sum/5
}


model_selection_gb <- rbind(model_selection_gb,model_selection_gb2)
# saveRDS(model_selection_gb,"model_selection_gb.rds")

model_selection_gb <- as.data.table(model_selection_gb)
model_selection_gb[,K_smote1:=factor(K_smote,levels = c(0,3,4,5,6),labels = c("no SMOTE","SMOTE(K=3)","SMOTE(K=4)","SMOTE(K=5)","SMOTE(K=6)"))]

p1 <- ggplot(model_selection_gb)+
  geom_point(aes(x=num_of_tree,y=cv_error,color=K_smote1))+
  geom_line(aes(x=num_of_tree,y=cv_error,color=K_smote1))+
  scale_x_continuous(breaks = seq(100,800,by=100))+
  theme_bw()+
  labs(x="number of trees",y="average error rate in 5-fold cross validation",color="choices of K in SMOTE",title = "hyperparameter tuning in gradient boost")

ggsave("gd_hyper_selection.pdf",p1,w=8,h=6)

# choose K=6, number of trees equals to 800

##### SVM #####
model_selection_svm <- expand.grid(model="SVM",K_smote=3:6,C=c(1:2,10:11,20:21),cv_error=0)
for (model in 1:dim(model_selection_svm)[1]){
  K1 <- model_selection_svm[model,]$K_smote
  C1 <- model_selection_svm[model,]$C
  train2 <- smotenc(train1,var="outcome",k=K1)
  svm_model1 <- ksvm(outcome~.,data=train2,type="C-svc",kernel="rbfdot",scaled = FALSE,cross=5,C=C1)
  model_selection_svm[model,]$cv_error <- svm_model1@cross
}

model_selection_svm1 <- expand.grid(model="SVM",K_smote=0,C=c(1:2,10:11,20:21),cv_error=0)

name_weight <- c(dim(train1[which(train1$outcome=="True"),])[1]/dim(train1)[1],dim(train1[which(train1$outcome=="False"),])[1]/dim(train1)[1])
names(name_weight) <- c("False","True")

for (model in 1:dim(model_selection_svm1)[1]){
  C1 <- model_selection_svm1[model,]$C
  svm_model1 <- ksvm(outcome~.,data=train1,type="C-svc",kernel="rbfdot",scaled = FALSE,cross=5,C=C1,class.weights=name_weight)
  model_selection_svm1[model,]$cv_error <- svm_model1@cross
}
model_selection_svm <- rbind(model_selection_svm1,model_selection_svm)
#saveRDS(model_selection_svm,"model_selection_svm.rds")
model_selection_svm <- as.data.table(model_selection_svm)
model_selection_svm[,K_smote1:=factor(K_smote,levels = c(0,3,4,5,6),labels = c("no SMOTE","SMOTE(K=3)","SMOTE(K=4)","SMOTE(K=5)","SMOTE(K=6)"))]

p1 <- ggplot(model_selection_svm)+
  geom_point(aes(x=C,y=cv_error,color=K_smote1))+
  geom_line(aes(x=C,y=cv_error,color=K_smote1))+
  scale_x_continuous(breaks = seq(1,21,by=2))+
  theme_bw()+
  labs(x="C in regularization term",y="average error rate in 5-fold cross validation",color="choices of K in SMOTE",title = "hyperparameter tuning in SVM")

# should choose C=21,K=3
ggsave("svm_hyper_selection.pdf",p1,w=8,h=6)
#### model comparison ####
data5 <- readRDS("data_after_feature_selection.rds")
data5 <- as.data.frame(data5)
train_index <- createDataPartition(data5$outcome,p=0.8)
train1 <- data5[train_index$Resample1,]
valid1 <- data5[-train_index$Resample1,]
summary(train1)
summary(valid1)

train2 <- smotenc(train1,var="outcome",k=3)
rf_model1 <- ranger(outcome~.,data=train2,importance = "impurity",probability = TRUE,num.trees=400)
rf_pre1 <- predict(rf_model1,data=valid1[,-22])
auc(response=valid1$outcome,predictor=rf_pre1$predictions[,1])
# 0.7607
rf_roc1 <- roc(response=valid1$outcome,predictor=rf_pre1$predictions[,1])
rf_t1 <- coords(rf_roc1,"best", ret = "threshold")$threshold
rf_pre2 <- data.table(true_label = valid1$outcome, pre_value = rf_pre1$predictions[,1])
set(rf_pre2,which(rf_pre2[,pre_value>rf_t1]),"pre_label","False")
set(rf_pre2,which(rf_pre2[,pre_value<=rf_t1]),"pre_label","True")


train3 <- smotenc(train1,var="outcome",k=6)
train3 <- as.data.table(train3)
set(train3,NULL,"outcome",train3[,as.integer(outcome)])
set(train3,NULL,"outcome",train3[,outcome-1])

gb_model1 <- gbm(outcome~.,data=train3,distribution="bernoulli",n.trees = 800)
gb_pre1 <- predict(gb_model1,newdata = valid1[,-22],n.trees = 800,type="response")
auc(response=valid1$outcome,predictor=gb_pre1)
# 0.7601
gb_roc1 <- roc(response=valid1$outcome,predictor=gb_pre1)
gb_t1 <- coords(gb_roc1,"best", ret = "threshold")$threshold

gb_pre2 <- data.table(true_label = valid1$outcome, pre_value = gb_pre1)
set(gb_pre2,which(gb_pre2[,pre_value>gb_t1]),"pre_label","True")
set(gb_pre2,which(gb_pre2[,pre_value<=gb_t1]),"pre_label","False")



train2 <- smotenc(train1,var="outcome",k=3)
svm_model1 <- ksvm(outcome~.,data=train2,type="C-svc",kernel="rbfdot",scaled = FALSE,C=21)
svm_pre1 <- predict(svm_model1,newdata = valid1[,-22])
svm_pre2 <- data.table(true_label = valid1$outcome, pre_label = svm_pre1)


saveRDS(svm_model1,"svm_model2.rds")
saveRDS(rf_model1,"rf_model2.rds")
saveRDS(gb_model1,"gb_model2.rds")

p1 <- ggplot(NULL)+
  geom_line(aes(x=1-rf_roc1$specificities,y=rf_roc1$sensitivities))+
  geom_point(aes(x=1-rf_roc1$specificities[which(rf_roc1$thresholds==rf_t1)],y=rf_roc1$sensitivities[which(rf_roc1$thresholds==rf_t1)]),color="blue")+
  annotate("text", x=0.15, y=0.75, label= "threshold",color="blue")+
  geom_line(aes(x=c(0,1),y=c(0,1)),linetype = "dashed")+
  theme_bw()+
  labs(x="1-specificity",y="sensitivity",title="ROC of RF model")


p2 <- ggplot(NULL)+
  geom_line(aes(x=1-gb_roc1$specificities,y=gb_roc1$sensitivities))+
  geom_point(aes(x=1-gb_roc1$specificities[which(gb_roc1$thresholds==gb_t1)],y=gb_roc1$sensitivities[which(gb_roc1$thresholds==gb_t1)]),color="blue")+
  annotate("text", x=0.25, y=0.8, label= "threshold",color="blue")+
  geom_line(aes(x=c(0,1),y=c(0,1)),linetype = "dashed")+
  theme_bw()+
  labs(x="1-specificity",y="sensitivity",title="ROC of GB model")

p3 <- ggarrange(p1,p2,ncol = 2,labels = c("A","B"))

ggsave("ROC.pdf",p3,w=10,h=5)


svm_pre3 <- svm_pre2[,list(count=.N),by=c("true_label","pre_label")]
svm_pre3
svm_pre4 <- data.table(model="svm",auc=NA,a=57,b=208,c=611,d=6406)


rf_pre3 <- rf_pre2[,list(count=.N),by=c("true_label","pre_label")]
rf_pre3
rf_pre4 <- data.table(model="rf",auc=0.7607,a=174,b=91,c=1662,d=5355)

gb_pre3 <- gb_pre2[,list(count=.N),by=c("true_label","pre_label")]
gb_pre3
gb_pre4 <- data.table(model="gb",auc=0.7601,a=196,b=69,c=2300,d=4717)


model_pre <- rbind(rf_pre4,gb_pre4,svm_pre4)
model_pre[,accuracy:=(a+d)/(a+b+c+d)]
model_pre[,precision:=a/(a+c)]
model_pre[,recall:=a/(a+b)]
model_pre[,specificity:=d/(c+d)]
model_pre[,sensitivity:=a/(a+b)]
model_pre[,F_measure:=(2*precision*recall)/(precision+recall)]

saveRDS(model_pre,"model_pre.rds")
model_pre <- readRDS("model_pre.rds")

