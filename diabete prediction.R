data = read.csv('/Users/ziyiyan/Downloads/diabetes-dataset.csv')
head(data)
library(tidyverse)
library(caret)
library(randomForest)
which(is.na(data))
columns_with_zeros <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
data[columns_with_zeros] <- lapply(data[columns_with_zeros], function(x) replace(x, x == 0, NA))
for (col in columns_with_zeros) {
  data[[col]] <- ifelse(is.na(data[[col]]), median(data[[col]], na.rm = TRUE), data[[col]])
}
summary(data)
correlation_matrix <- cor(data)
print(correlation_matrix)
library(ggplot2)
library(gridExtra)
hist_plots <- lapply(names(data), function(x) ggplot(data, aes_string(x)) + 
                       geom_histogram(bins = 30) + 
                       ggtitle(paste("Histogram of", x)))
grid.arrange(grobs = hist_plots, nrow = 3, ncol = 3)
scatter <- ggplot(data, aes(x = Glucose, y = Age, color = as.factor(Outcome))) + geom_point(size=0.5) +
  labs(title = "Scatter plot of Glucose vs Age")
print(scatter)
scatter <- ggplot(data, aes(x = Glucose, y = Pregnancies, color = as.factor(Outcome))) + geom_point(size=0.4) +
  labs(title = "Scatter plot of Glucose vs Pregnancies")
print(scatter)
scatter <- ggplot(data, aes(x = Glucose, y = Insulin, color = as.factor(Outcome))) + geom_point(size=0.4) +
  labs(title = "Scatter plot of Glucose vs Insulin")

print(scatter)
cols <- names(data)[1:(ncol(data)-1)]  
plot_list <- list()
for (col in cols) {
  p <- (ggplot(data, aes(x = as.factor(Outcome), y = get(col))) +
          geom_boxplot() +
          labs(title = paste("Boxplot of", col, "by Outcome"), x = "Outcome", y = col))
  plot_list[[col]] <- p
}
grid.arrange(grobs = plot_list, nrow = 2, ncol = 4)

train_columns <- sapply(data, is.numeric)
train_columns["Outcome"] <- FALSE
data[train_columns] <- scale(data[train_columns])
set.seed(123)
train.prop <- 0.80
strats <- data$Outcome
rr <- split(1:length(strats), strats)
idx <- sort(as.numeric(unlist(sapply(rr, 
                                     function(x) sample(x, length(x)*train.prop)))))
data.train <- data[idx, ]
data.test <- data[-idx, ]
full.logit <- glm(Outcome ~ . ,data = data.train, 
                  family = binomial(link = "logit"))
summary(full.logit)
null.logit <- glm(Outcome ~ 1, data = data.train, 
                  family = binomial(link = "logit"))
summary(null.logit)
both.logit <- step(null.logit, list(lower = formula(null.logit),
                                    upper = formula(full.logit)),
                   direction = "both", trace = 0, data = data.train)
formula(both.logit)
summary(both.logit)
library(pROC)
pred.both <- predict(both.logit, newdata = data.train, type = "response")
(table.both <- table(pred.both > 0.5, data.train$Outcome))
(accuracy.both <- round((sum(diag(table.both))/sum(table.both))*100, 3)) 
roc.both <- roc(data.train$Outcome ~ pred.both, plot = TRUE, 
                legacy.axes = TRUE, print.auc = TRUE)
pred.both.test <- predict(both.logit, newdata = data.test, type = "response")
(table.both.test <- table(pred.both.test > 0.5, data.test$Outcome))
(accuracy.both.test <- round((sum(diag(table.both.test))/sum(table.both.test))*100, 3)) 
roc.both.test <- roc(data.test$Outcome ~ pred.both.test, plot = TRUE, 
                     legacy.axes = TRUE, print.auc = TRUE)
variables <- train_columns
interaction.var <-
  combn(variables, 2,
        FUN = function(x)
          paste(x, collapse = ":")
  )
formula.inter <-
  as.formula(paste("Outcome ~ . +", paste(interaction.var, collapse = "+")))
all.logit <- glm(formula = formula.inter,
                 family = binomial(link = "logit"),
                 data = data.train)
summary(all.logit )
pred.all <- predict(all.logit, newdata = data.train, type = "response")
(table.all <- table(pred.all > 0.5, data.train$Outcome))
(accuracy.all <- round((sum(diag(table.all))/sum(table.all))*100, 3)) 
roc.all <- roc(data.train$Outcome ~ pred.all, plot = TRUE, 
               legacy.axes = TRUE, print.auc = TRUE)
pred.all.test <- predict(all.logit, newdata = data.test, type = "response")
(table.all.test <- table(pred.all.test > 0.5, data.test$Outcome))
(accuracy.all.test <- round((sum(diag(table.all.test))/sum(table.all.test))*100, 3)) 
roc.all.test <- roc(data.test$Outcome ~ pred.all.test, plot = TRUE, 
                    legacy.axes = TRUE, print.auc = TRUE)
full.probit <- glm(Outcome ~ . ,data = data.train , 
                   family = binomial(link = "probit"))
summary(full.probit)
null.probit <- glm(Outcome ~ 1, data = data.train, 
                   family = binomial(link = "probit"))
summary(null.probit)
both.probit <- step(null.probit, list(lower = formula(null.probit),
                                      upper = formula(full.probit)),
                    direction = "both", trace = 0, data = data.train)
formula(both.probit)
summary(both.probit)
pred.both.probit <- predict(both.probit, newdata = data.train, type = "response")
(table.both.probit <- table(pred.both.probit > 0.5, data.train$Outcome))
(accuracy.both.probit <- round((sum(diag(table.both.probit))/sum(table.both.probit))*100, 3)) 
roc.both.probit <- roc(data.train$Outcome ~ pred.both.probit, plot = TRUE, 
                       legacy.axes = TRUE, print.auc = TRUE)
pred.both.probit.test <- predict(both.probit, newdata = data.test, type = "response")
(table.both.probit.test <- table(pred.both.probit.test > 0.5, data.test$Outcome))
(accuracy.both.probit.test <- round((sum(diag(table.both.probit.test))/sum(table.both.probit.test))*100, 3)) 
roc.both.probit.test <- roc(data.test$Outcome ~ pred.both.probit.test, plot = TRUE, 
                            legacy.axes = TRUE, print.auc = TRUE)
rf_model <- randomForest(Outcome ~ Glucose + Pregnancies + BMI + DiabetesPedigreeFunction + Age, 
                         data = data.train, 
                         ntree = 500, 
                         importance = TRUE) 
print(rf_model)
pred.rf <- predict(rf_model, newdata = data.train, type = "response")
(table.rf <- table(pred.rf > 0.5, data.train$Outcome))
(accuracy.rf <- round((sum(diag(table.rf))/sum(table.rf))*100, 3)) 
roc.rf <- roc(data.train$Outcome ~ pred.rf, plot = TRUE, 
              legacy.axes = TRUE, print.auc = TRUE)
pred.rf.test <- predict(rf_model, newdata = data.test, type = "response")
(table.rf.test <- table(pred.rf.test > 0.5, data.test$Outcome))
(accuracy.rf.test <- round((sum(diag(table.rf.test))/sum(table.rf.test))*100, 3)) 
roc.rf.test <- roc(data.test$Outcome ~ pred.rf.test, plot = TRUE, 
                   legacy.axes = TRUE, print.auc = TRUE)
