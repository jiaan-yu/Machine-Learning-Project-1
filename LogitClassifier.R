## Logistic regression classifier

library(glmnet)
library(leaps)
library(pROC)

# Read feature files and covert them into matrics
data_ = read.table("training-features.txt", sep = ",",header = T)
dev = read.table("development-features.txt", sep = ",",header = T)
test = read.table("test-features.txt", sep = ",",header = T)
Xtrain = as.matrix(data_[,-c(1, 40)])
Ytrain= c(rep(1, 20000), rep(0, 20000))
Xdev = as.matrix(dev[,-c(1, 40)])
Ydev = c(rep(1, 1000), rep(0, 1000))
Xtest = as.matrix(test[,-c(1, 40)])


# Build a logistic regression classifier (full model)
new_logit <- glm(label ~ . - Id, data = data_, family = "binomial"(link = "logit"), maxit=2000)
pred = predict(new_logit, dev, type = "response")
dev_ROC = roc(dev$label, pred, direction="<")
# AUC of development data is 0.98
dev_ROC$auc
# Make predictions for test file
Prediction = predict(new_logit, test, type = "response")
write.csv(x = cbind(id, predictions), file = "newpredict_R3.csv",row.names = F)


# Forward stepwise selection
lm7.fit <- regsubsets(Xtrain, Ytrain, method="forward", nvmax=20, best=1)
plot(summary(lm7.fit)$rss, xlab="Number of predicto
     rs", ylab="RSS")
# Choose the first 12 features
# f6, f14, f15, f16, f18, f19, f20, f21, f22, f32, f36, f38
lm7_coef <- coef(lm7.fit, 12)
# Test on the develpemnt data
# AUC 0.92
Xdev_new <- as.matrix(cbind(1, Xdev))
colnames(Xdev_new)[1] <- "(Intercept)"
Ydev_new <- Xdev_new[, names(lm7_coef)] %*% as.matrix(lm7_coef)
dev_ROC_new = roc(dev$label, as.numeric(Ydev_new), direction="<")
# Generate test data
# Scores 0.85 on test data
Xtest_new <- as.matrix(cbind(1, Xtest))
colnames(Xtest_new)[1] <- "(Intercept)"
predictions <- Xtest_new[, names(lm7_coef)] %*% as.matrix(lm7_coef)
write.csv(x = predictions, file = "predict_forward_selection.csv",row.names = F, col.names = F)

# LASSO regulasation
lambda.opt <- cv.glmnet(as.matrix(Xtrain), Ytrain,
                        family = "binomial",
                        type.measure = "class",
                        nfolds = 10,
                        alpha = 1)$lambda.min
lasso.fit <- glmnet(as.matrix(Xtrain), Ytrain,
                    family = "binomial",
                    lambda = lambda.opt,
                    alpha = 1)
Y3.predict <- predict(lasso.fit, as.matrix(Xtest), s=c("lambda.min"), 
                      type = "class")
#head(Y3.predict)
write.csv(x = Y3.predict, file = "predict_lasso.csv",row.names = F, col.names = F)