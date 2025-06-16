# load packages
library(Hmisc)
library(car)
library(rms) # can implement logistic regression model (lrm)
library(pROC)
library(rmda)
library(dplyr)
library(readxl);library(caret);library(glmnet);library(corrplot)
library(Metrics);library(ggplot2)
# packages needed for result merging
library(plyr)
# packages for logistic regression
library(epiDisplay)# quickly output OR, 95% CI, P
library(gtsummary)# beautiful three-line table
library("writexl")
library(nsROC)
library(caret)
library(ggDCA)
library("pheatmap")

# 1. set path
# set path
project_path <- "...\\grace_icu\\" # change to your project path
path = 'D:/project/older_kg_nlp/final_revision/'
num_dir <- paste(project_path, "data\\use\\seq_240\\",sep = "")
text_split_dir <- paste(project_path,"result\\subtext\\",sep = "")
output_dir <- text_split_dir
output_dir_new <- paste(text_split_dir,"roc_result/",sep = "")
if (dir.exists(output_dir)){  
  print("The direcoty exists")
}else{  
  # create
  dir.create(output_dir)
}
if (dir.exists(output_dir_new)){  
  print("The direcoty exists")
}else{  
  # create
  dir.create(output_dir_new)
}

# 2. load data and merge note value and numeric values - merge train and val together
development_num_older <- read.csv(paste(num_dir,"development_num_older.csv",sep = ""), header=TRUE)
temporal_num_older <- read.csv(paste(num_dir,"temporal_num_older.csv",sep = ""), header=TRUE)
external_num_older <- read.csv(paste(num_dir,"external_num_older.csv",sep = ""), header=TRUE)
development_num_older <- subset(development_num_older, select = -c(subject_id, hadm_id, inr_min, inr_max, braden_flag, braden_score_cat))
temporal_num_older <- subset(temporal_num_older, select = -c(subject_id, hadm_id, inr_min, inr_max, braden_flag, braden_score_cat))
external_num_older <- subset(external_num_older, select = -c(inr_min, inr_max, braden_flag, braden_score_cat))

type_list <- list("", "_chief_complaint", "_history_of_present_illness", 
                  "_medications_on_admission", "_past_medical_history", "_physical_exam")

for(t_type in type_list){
  ## step 2
  text_older <- read.csv(paste(text_split_dir, "text_all", t_type, ".csv",sep = ""), header=TRUE)
  #    [1] extract needed columns
  text_older <- subset(text_older, select = c(all_patient_ids, probs_1, db_type))
  #    [2] change to the suitable names
  text_older <- plyr::rename(text_older, c("all_patient_ids"="id", "probs_1"="preICU_risk_score"))
  text_older$preICU_risk_score_raw <- text_older$preICU_risk_score
  #     [3] merge dataframes and merge train and validation data
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
  #      [4] remove no need categorical columns
  training_dataset <- subset(training_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
  test_dataset <- subset(test_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
  temp_dataset <- subset(temp_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
  ext_dataset <- subset(ext_dataset, select = -c(first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
  # #      [5] remove multiple vasopressors
  # training_dataset <- subset(training_dataset, select = -c(dobutamine,dopamine,norepinephrine,epinephrine))
  # test_dataset <- subset(test_dataset, select = -c(dobutamine,dopamine,norepinephrine,epinephrine))
  # temp_dataset <- subset(temp_dataset, select = -c(dobutamine,dopamine,norepinephrine,epinephrine))
  #      [6] factor variables setting
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
  
  # step 2
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
    
    f_lrm_num_text <-lrm(label ~ preICU_risk_score + shock_index + code_status + activity_bed + vent + 
                           lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score
                         , data=training_dataset, x=TRUE, y=TRUE)
    
    #    [0] get no text model
    f_lrm_num <-lrm(label ~ shock_index + code_status + activity_bed + vent + 
                      lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score
                    , data=training_dataset, x=TRUE, y=TRUE)
    #    [1] predict
    pred_f_num_text <- predict(f_lrm_num_text, data_plot)
    pred_f_num_text <- 1/(1+exp(-pred_f_num_text))
    pred_f_num <- predict(f_lrm_num, data_plot)
    pred_f_num <- 1/(1+exp(-pred_f_num))
    # modelroc <- roc(data_plot$label,pred_f_num_text,ci = TRUE)
    #    [2] get comparison dataframe
    roc_test_set <- subset(data_plot, select = c(id, label, preICU_risk_score_raw, sofa, saps))
    roc_test_set$pred_f_num_text <- pred_f_num_text
    roc_test_set$pred_f_num <- pred_f_num
    write.csv(roc_test_set, paste(output_dir_new, "roc_", i, t_type, "_set.csv",sep = ""))
    
    #    [3] get roc values
    roc.pred1 <- roc(roc_test_set$label, roc_test_set$pred_f_num_text, percent = TRUE, main = "Smoothing")
    roc.pred2 <- roc(roc_test_set$label, roc_test_set$pred_f_num, percent = TRUE, main = "Smoothing")
    roc.pred3 <- roc(roc_test_set$label, roc_test_set$preICU_risk_score_raw, percent = TRUE, main = "Smoothing")
    roc.pred4 <- roc(roc_test_set$label, roc_test_set$saps, percent = TRUE, main = "Smoothing")
    roc.pred5 <- roc(roc_test_set$label, roc_test_set$sofa, percent = TRUE, main = "Smoothing")
    #    [4] plot roc curves comparisons
    png(file=paste(output_dir_new, "roc_", i, t_type, ".png", sep = ""), width=1900, height=1600, res = 300)
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
    # legend1 <- paste("GRACE-ICU (", round(roc_95_pred1[1],1), round(roc_95_pred1[2],1), sep = " ")
    legend("bottomright", 
           legend = c(
             paste("GRACE-ICU (AUC = ", round(roc.pred1$auc[1],1)/100, ")", sep = ""), 
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
}