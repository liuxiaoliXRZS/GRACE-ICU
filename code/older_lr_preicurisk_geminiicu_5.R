# 0. load all needed data
# 1. nomogram plot
# 2. train multivariable model
# Major targets
#    3. compare performance in the internal and temporal set -- ROC curves
# A more intuitive comparison, but the final results and charts of the model will be plotted in the subsequent code set.
#    4. calibration plot
#    5. DCA curves


# load packages
rm(list = ls()) # Clear all data in R
library(Hmisc)
library(car)
library(rms)# Can implement logistic regression model (lrm)
library(pROC)
library(nsROC)
library(rmda)
library(dplyr)
library(readxl);library(caret);library(glmnet);library(corrplot)
library(Metrics);library(ggplot2)
# Packages needed for result merging
library(plyr)
# Packages needed for result merging
library(epiDisplay)# Quickly output OR, 95% CI, P
library(gtsummary)# Beautiful three - line table
library(ggDCA)
library("writexl")
library("pheatmap")
library("corrplot")

# set path
project_path <- "...\\gemini_icu\\" # change to your project path
num_dir <- paste(project_path, "data\\use\\seq_240\\",sep = "")
text_dir <- paste(project_path, "result\\note\\clinical_longformer\\seq_240\\max_len-512_bs-16_epoch-2_lr-1e-05\\",sep = "")
output_dir <- paste(project_path, "result\\note_num\\nomogram\\",sep = "")


#                 ------------------------- Step 0 -------------------------                   #
#
# -------------------------------------------------------------------------------------------- #
development_num_older <- read.csv(paste(num_dir,"development_num_older.csv",sep = ""), header=TRUE)
temporal_num_older <- read.csv(paste(num_dir,"temporal_num_older.csv",sep = ""), header=TRUE)
external_num_older <- read.csv(paste(num_dir,"external_num_older.csv",sep = ""), header=TRUE)
development_num_older <- subset(development_num_older, select = -c(subject_id, hadm_id, inr_min, inr_max, braden_flag, braden_score_cat))
temporal_num_older <- subset(temporal_num_older, select = -c(subject_id, hadm_id, inr_min, inr_max, braden_flag, braden_score_cat))
external_num_older <- subset(external_num_older, select = -c(inr_min, inr_max, braden_flag, braden_score_cat))
text_older <- read.csv(paste(text_dir, "text_all.csv",sep = ""), header=TRUE)
#    [1] extract needed columns
text_older <- subset(text_older, select = c(all_patient_ids, probs_1, db_type))
#    [2] change to the suitable names
text_older <- plyr::rename(text_older, c("all_patient_ids"="id", "probs_1"="preICU_risk_score"))
text_older$preICU_risk_score_raw <- text_older$preICU_risk_score
#    [3] merge dataframes and merge train and validation data
training_dataset <- merge(
  x = subset(text_older[text_older$db_type == 'train',], select = c(id, preICU_risk_score, preICU_risk_score_raw)), 
  y = development_num_older, by = "id")
training_dataset$preICU_risk_score <- round(10*training_dataset$preICU_risk_score)  # more appropriate to expand the data by 10 times
validation_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'val',], select = c(id, preICU_risk_score, preICU_risk_score_raw)), 
  y = development_num_older, by = "id")
validation_dataset$preICU_risk_score <- round(10*validation_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
test_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'test',], select = c(id, preICU_risk_score, preICU_risk_score_raw)), 
  y = development_num_older, by = "id")
test_dataset$preICU_risk_score <- round(10*test_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
temp_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'temp',], select = c(id, preICU_risk_score, preICU_risk_score_raw)), 
  y = temporal_num_older, by = "id")
temp_dataset$preICU_risk_score <- round(10*temp_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
ext_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'ext',], select = c(id, preICU_risk_score, preICU_risk_score_raw)), 
  y = external_num_older, by = "id")  
ext_dataset$preICU_risk_score <- round(10*ext_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
training_dataset <- rbind(training_dataset, validation_dataset) # combine train and validation set
#     [4] remove no need categorical columns
training_dataset <- subset(training_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
test_dataset <- subset(test_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
temp_dataset <- subset(temp_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
ext_dataset <- subset(ext_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
#     [5] factor variables setting
cols <- c(
  "label", "code_status", "code_status_eva_flag", "activity_stand", "vent", 
  "gender", "activity_bed", 
  "activity_sit", "activity_eva_flag", "pao2fio2ratio_vent_flag", 
  "pao2fio2ratio_novent_flag", "bilirubin_max_flag", "alp_max_flag", "alt_max_flag", 
  "ast_max_flag", "baseexcess_min_flag" , "fio2_max_flag", "lactate_max_flag", "lymphocytes_max_flag", 
  "lymphocytes_min_flag", "neutrophils_min_flag", "paco2_max_flag", 
  "pao2_min_flag", "nlr_flag", "vasopressor", "rrt", "gnri_flag"
)
training_dataset[cols] <- lapply(training_dataset[cols], factor)  ## as.factor() could also be used
test_dataset[cols] <- lapply(test_dataset[cols], factor)  ## as.factor() could also be used
temp_dataset[cols] <- lapply(temp_dataset[cols], factor)  ## as.factor() could also be used
ext_dataset[cols] <- lapply(ext_dataset[cols], factor)  ## as.factor() could also be used
write.csv(training_dataset, paste(output_dir, "training_dataset.csv",sep = ""))
write.csv(test_dataset, paste(output_dir, "test_dataset.csv",sep = ""))
write.csv(temp_dataset, paste(output_dir, "temp_dataset.csv",sep = ""))
write.csv(ext_dataset, paste(output_dir, "ext_dataset.csv",sep = ""))
rm(development_num_older, temporal_num_older, external_num_older, validation_dataset, text_older, text_dir)

print(paste0("training set rows",dim(training_dataset)[1]," training set columns",dim(training_dataset)[2])) # dim(x) #查看行列数
print(paste0("test set rows",dim(test_dataset)[1]," test set columns",dim(test_dataset)[2]))
print(paste0("temp set rows",dim(temp_dataset)[1]," temp set columns",dim(temp_dataset)[2]))
summary(training_dataset) # Check the distribution of the data
summary(test_dataset)
summary(temp_dataset)



#                 ------------------------- Step 1 -------------------------                   #
#
# -------------------------------------------------------------------------------------------- #
#       [1] plot correlation of focused features
# "gcs_min","cci_score","admission_type","shock_index","vent","activity_bed","code_status","vasopressor","resp_rate_mean","preICU_risk_score"
x <- subset(training_dataset, select = c(preICU_risk_score, cci_score, gcs_min, shock_index, resp_rate_mean, lactate_max, inr_min, temperature_mean))
# column with column co variation
cov(x)
pheatmap::pheatmap(cov(x)) # Heat Map
# correlation
cor(x)
pheatmap::pheatmap(cor(x))  # Heat Map near to zero == less correlation
corrplot(cor(x))
png(filename=paste(output_dir,"corrplot.png",sep = ""), width = 1600, height = 1600, res = 300)
corrplot(cor(x),method="color",addCoef.col="grey") # Display with color, show correlation coefficient
dev.off()
rm(x)

#       [2] nomogram
f_lrm_num_text <-lrm(label ~ preICU_risk_score + shock_index + code_status + activity_bed + vent + 
                       lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score
                     , data=training_dataset, x=TRUE, y=TRUE) #
print(f_lrm_num_text)
ddist <- datadist(training_dataset)
options(datadist='ddist')
png(file=paste(output_dir, "nomogram_num_text.png", sep = ""), width=2200, height=2300, res = 300)
nomogram <- nomogram(f_lrm_num_text,fun=function(x)1/(1+exp(-x)),
                     fun.at = c(0.01,0.1,0.3,0.5,0.8,0.9,0.99),
                     funlabel = "Probability of death",
                     lp=F,
                     conf.int = F,
                     abbrev = F)
plot(nomogram)
dev.off()
svg(file=paste(output_dir, "nomogram_num_text.svg", sep = ""), width=11, height=9)
nomogram <- nomogram(f_lrm_num_text,fun=function(x)1/(1+exp(-x)),
                     fun.at = c(0.01,0.1,0.3,0.5,0.8,0.9,0.99),
                     funlabel = "Probability of death",
                     lp=F,
                     conf.int = F,
                     abbrev = F)
plot(nomogram)
dev.off()
rm(ddist, nomogram)


#                 ------------------------- Step 2 save analyze results-------------------------                 #
#
# -------------------------------------------------------------------------------------------------------------- #
loop_list <- list("test", "temp", "ext") 
for(i in loop_list){
  if(i=='test'){
    data_plot <- test_dataset
  }
  else if(i=='temp'){
    data_plot <- temp_dataset    
  }
  else{
    data_plot <- ext_dataset
  }
  #    [0] get no text model
  f_lrm_num <-lrm(label ~ shock_index + code_status + activity_bed + vent + 
                    lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score, data=training_dataset, x=TRUE, y=TRUE)
  #    [1] predict
  pred_f_num_text <- predict(f_lrm_num_text, data_plot)
  pred_f_num_text <- 1/(1+exp(-pred_f_num_text))
  pred_f_num <- predict(f_lrm_num, data_plot)
  pred_f_num <- 1/(1+exp(-pred_f_num))
  # modelroc <- roc(data_plot$label,pred_f_num_text,ci = TRUE)
  #    [2] get comparison dataframe
  roc_test_set <- subset(data_plot, select = c(id, label, preICU_risk_score_raw, sofa, saps))
  # roc_test_set$preICU_risk_score <- roc_test_set$preICU_risk_score/10
  roc_test_set$pred_f_num_text <- pred_f_num_text
  roc_test_set$pred_f_num <- pred_f_num
  write.csv(roc_test_set, paste(output_dir, "roc_", i, "_set.csv",sep = ""))

  #    [3] get roc values
  roc.pred1 <- roc(roc_test_set$label, roc_test_set$pred_f_num_text, percent = TRUE, main = "Smoothing")
  roc.pred2 <- roc(roc_test_set$label, roc_test_set$pred_f_num, percent = TRUE, main = "Smoothing")
  roc.pred3 <- roc(roc_test_set$label, roc_test_set$preICU_risk_score_raw, percent = TRUE, main = "Smoothing")
  roc.pred4 <- roc(roc_test_set$label, roc_test_set$saps, percent = TRUE, main = "Smoothing")
  roc.pred5 <- roc(roc_test_set$label, roc_test_set$sofa, percent = TRUE, main = "Smoothing")
  #    [4] plot roc curves comparisons
  png(file=paste(output_dir, "roc_", i, ".png", sep = ""), width=1900, height=1600, res = 300)
  plot.roc(roc_test_set$label, roc_test_set$pred_f_num_text, percent = TRUE, add =  FALSE, asp = NA, cex.axis = 1.2, cex.lab = 1.5, col = "blue")
  lines(roc.pred2, type = "l", lty = 1, col = "red")
  lines(roc.pred3, type = "l", lty = 1, col = "green")
  lines(roc.pred4, type = "l", lty = 1, col = "yellow")
  lines(roc.pred5, type = "l", lty = 1, col = "purple")
  # #    [5] get the 95% CI values
  # set.seed(1234)
  # roc_95_pred1 <- ci.auc(roc.pred1, conf.level=0.95, method=c("bootstrap"), boot.n = 500, boot.stratified = TRUE)
  # set.seed(1234)
  # roc_95_pred2 <- ci.auc(roc.pred2, conf.level=0.95, method=c("bootstrap"), boot.n = 500, boot.stratified = TRUE)
  # set.seed(1234)
  # roc_95_pred3 <- ci.auc(roc.pred3, conf.level=0.95, method=c("bootstrap"), boot.n = 500, boot.stratified = TRUE)
  # set.seed(1234)
  # roc_95_pred4 <- ci.auc(roc.pred4, conf.level=0.95, method=c("bootstrap"), boot.n = 500, boot.stratified = TRUE)
  # set.seed(1234)
  # roc_95_pred5 <- ci.auc(roc.pred5, conf.level=0.95, method=c("bootstrap"), boot.n = 500, boot.stratified = TRUE)
  # legend1 <- paste("GEMINI-ICU (", round(roc_95_pred1[1],1), round(roc_95_pred1[2],1), sep = " ")
  legend("bottomright", 
        legend = c(
          paste("GEMINI-ICU (AUC = ", round(roc.pred1$auc[1],1)/100, ")", sep = ""), 
          paste("Structured data score (AUC = ", round(roc.pred2$auc[1],1)/100, ")", sep = ""), 
          paste("preICU_risk_score (AUC = ", round(roc.pred3$auc[1],1)/100, ")", sep = ""), 
          paste("SAPSII (AUC = ", round(roc.pred4$auc[1],1)/100, ")", sep = ""), 
          paste("SOFA (AUC = ", round(roc.pred5$auc[1],1)/100, ")", sep = "")
          ), 
        col = c("blue", "red", "green", "yellow", "purple"),
        lty = c(1, 1, 1, 1, 1), cex=1.3, pt.cex = 1, lwd=2) 
  dev.off()
  rm(roc.pred1, roc.pred2, roc.pred3, roc.pred4, roc.pred5, data_plot)
}
rm(loop_list, i, roc_test_set)





#                 ------------------------- Step 3 calibration plot -------------------------                  #
#
# ------------------------------------------------------------------------------------------------------------ #
cal <- calibrate(f_lrm_num_text, predy=seq(0, 1.0, length=50), B=500)#method=c("boot","crossvalidation",".632","randomization")
png(file=paste(output_dir, "calibration_train.png", sep = ""), width=1900, height=1600, res = 300) 
plot(cal, cex.axis = 1.2, cex.lab = 1.5)
dev.off()

pred.lg<- predict(f_lrm_num_text, temp_dataset)
temp_dataset$prob <- 1/(1+exp(-pred.lg))
png(file=paste(output_dir, "calibration_temp.png", sep = ""), width=1900, height=1600, res = 300)
val.prob(temp_dataset$prob, temp_dataset$label %>% recode("0" = 0, "1" = 1) %>% as.numeric(), cex=1)
dev.off()
rm(pred.lg)

pred.lg<- predict(f_lrm_num_text, test_dataset)
test_dataset$prob <- 1/(1+exp(-pred.lg))
png(file=paste(output_dir, "calibration_test.png", sep = ""), width=1900, height=1600, res = 300)
val.prob(test_dataset$prob, test_dataset$label %>% recode("0" = 0, "1" = 1) %>% as.numeric(), cex=1)
dev.off()
rm(pred.lg)

pred.lg<- predict(f_lrm_num_text, ext_dataset)
ext_dataset$prob <- 1/(1+exp(-pred.lg))
png(file=paste(output_dir, "calibration_ext.png", sep = ""), width=1900, height=1600, res = 300)
val.prob(ext_dataset$prob, ext_dataset$label %>% recode("0" = 0, "1" = 1) %>% as.numeric(), cex=1)
dev.off()
rm(pred.lg)


#                 ------------------------- Step 4 DCA analysis-------------------------                   #
#
# -------------------------------------------------------------------------------------------------------- #
loop_list <- list("test", "temp", "ext")
for(i in loop_list){
  if(i=='test'){
    training_dataset_dca <- test_dataset # training_dataset, test_dataset, temp_dataset
  }
  else if(i=='temp'){
    training_dataset_dca <- temp_dataset
  }
  else{
    training_dataset_dca <- ext_dataset
  }

  training_dataset_dca$label <- as.numeric(training_dataset_dca$label)-1
  model1 <- decision_curve(label ~ preICU_risk_score + shock_index + code_status + activity_bed + vent + 
                             lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score,
                          data=training_dataset_dca,
                          #thresholds = seq(0, .4, by = .01),
                          study.design = 'cohort',
                          bootstraps = 500)
  model2 <- decision_curve(label ~ shock_index + code_status + activity_bed + vent + 
                             lactate_max + braden_score + cci_score + resp_rate_mean + gcs_min + temperature_mean,
                          data=training_dataset_dca,
                          #thresholds = seq(0, .4, by = .01),
                          study.design = 'cohort',
                          bootstraps = 500)
  model3 <- decision_curve(label ~ preICU_risk_score,
                          data=training_dataset_dca,
                          #thresholds = seq(0, .4, by = .01),
                          study.design = 'cohort',
                          bootstraps = 500)
  model4 <- decision_curve(label ~ saps,
                          data=training_dataset_dca,
                          #thresholds = seq(0, .4, by = .01),
                          study.design = 'cohort',
                          bootstraps = 500)
  model5 <- decision_curve(label ~ sofa,
                          data=training_dataset_dca,
                          #thresholds = seq(0, .4, by = .01),
                          study.design = 'cohort',
                          bootstraps = 500)
  # Remove the confidence interval curve
  png(file=paste(output_dir, "dca_", i, ".png", sep = ""), width=1900, height=1600, res = 300)
  plot_decision_curve( list(model1, model2, model3, model4, model5),
                      curve.names = c("GEMINI-ICU", "Structured data score", "preICU_risk_score", "SAPSII", "SOFA"),
                      col = c("blue", "red", "green", "yellow", "purple"),
                      confidence.intervals = FALSE,  #remove confidence intervals
                      cost.benefit.axis = FALSE, #remove cost benefit axis
                      xlim = c(0, 1), #set xlim
                      legend.position = "topright") #remove the legend
  dev.off()
  rm(model1, model2, model3, model4, model5, training_dataset_dca)
}