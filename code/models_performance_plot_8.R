# load packages
rm(list = ls()) # Remove all data in R
library(Hmisc)
library(car)
library(rms) # Can implement logistic regression model (lrm)
library(pROC)
library(nsROC)
library(rmda)
library(dplyr)
library(readxl);library(caret);library(glmnet);library(corrplot)
library(Metrics);library(ggplot2)
# Packages needed for result merging
library(plyr)
# Packages for logistic regression
library(epiDisplay) # Quickly output OR, 95% CI, P
library(gtsummary) # Beautiful three-line tables
library(ggDCA)
library("writexl")
library("pheatmap")
library("corrplot")
library("PredictABEL")
library("nricens")

# set path
project_path <- "...\\grace_icu\\" # change to your project path
all_metrics_path <- paste(project_path, "result\\plot_need\\",sep = "")
roc_data_path <- paste(project_path, "result\\note_num\\nomogram\\",sep = "")
output_dir <- paste(project_path, "result\\plot_need\\",sep = "")



#                 ------------------------- Step 1 ROC plot -------------------------                  #
#
# ---------------------------------------------------------------------------------------------------- #
loop_list <- list("test", "temp", "ext") 
for(i in loop_list){
  roc_eva_set <- read.csv(paste(roc_data_path, "roc_", i, "_set.csv",sep = ""), header=TRUE)
  #    [1] get roc values
  roc.pred1 <- roc(roc_eva_set$label, roc_eva_set$pred_f_num_text, percent = TRUE, main = "Smoothing")
  roc.pred2 <- roc(roc_eva_set$label, roc_eva_set$pred_f_num, percent = TRUE, main = "Smoothing")
  roc.pred3 <- roc(roc_eva_set$label, roc_eva_set$preICU_risk_score_raw, percent = TRUE, main = "Smoothing")
  roc.pred4 <- roc(roc_eva_set$label, roc_eva_set$saps, percent = TRUE, main = "Smoothing")
  roc.pred5 <- roc(roc_eva_set$label, roc_eva_set$sofa, percent = TRUE, main = "Smoothing")
  #    [4] plot roc curves comparisons
  png(file=paste(output_dir, "roc_", i, ".png", sep = ""), width=1900, height=1600, res = 300)
  plot.roc(roc_eva_set$label, roc_eva_set$pred_f_num_text, percent = TRUE, add =  FALSE, asp = NA, cex.axis = 1.2, cex.lab = 1.5, col = "blue")
  lines(roc.pred2, type = "l", lty = 1, col = "red")
  lines(roc.pred3, type = "l", lty = 1, col = "green")
  lines(roc.pred4, type = "l", lty = 1, col = "yellow")
  lines(roc.pred5, type = "l", lty = 1, col = "purple")
  # #    [5] get the 95% CI values
  roc_need <- read.csv(paste(all_metrics_path, "all_models_performance_95ci.csv",sep = ""), header=TRUE)
  roc_need$roc_auc_use <- format(as.numeric(sub(" .*", "", roc_need$roc_auc)), nsmall = 3)
  roc_need <- roc_need[roc_need$db_type == i, ]  
  legend("bottomright", 
         legend = c(
           paste("GRACE-ICU (AUC = ", roc_need[roc_need$model_name == "pred_f_num_text",]$roc_auc_use, ")", sep = ""),           
           paste("Structured data score (AUC = ", roc_need[roc_need$model_name == "pred_f_num",]$roc_auc_use, ")", sep = ""), 
           paste("preICU risk score (AUC = ", roc_need[roc_need$model_name == "preICU_risk_score",]$roc_auc_use, ")", sep = ""), 
           paste("SAPSII (AUC = ", roc_need[roc_need$model_name == "saps",]$roc_auc_use, ")", sep = ""), 
           paste("SOFA (AUC = ", roc_need[roc_need$model_name == "sofa",]$roc_auc_use, ")", sep = "")
         ), 
         col = c("blue", "red", "green", "yellow", "purple"),
         lty = c(1, 1, 1, 1, 1), cex=1.3, pt.cex = 1, lwd=2) 
  dev.off()
  rm(roc.pred1, roc.pred2, roc.pred3, roc.pred4, roc.pred5, data_plot)
}
rm(loop_list, i, roc_eva_set, roc_need)


#                 ------------------------- Step 2 calibration plot -------------------------                  #
#
# ------------------------------------------------------------------------------------------------------------ #
data_need <- read.csv(paste(roc_data_path, "roc_test_set.csv",sep = ""), header=TRUE)
png(file=paste(output_dir, "calibration_test.png", sep = ""), width=1900, height=1600, res = 300)
val.prob(data_need$pred_f_num_text, data_need$label %>% recode("0" = 0, "1" = 1) %>% as.numeric(), cex=1)
dev.off()
rm(pred.lg)

data_need <- read.csv(paste(roc_data_path, "roc_temp_set.csv",sep = ""), header=TRUE)
png(file=paste(output_dir, "calibration_temp.png", sep = ""), width=1900, height=1600, res = 300)
val.prob(data_need$pred_f_num_text, data_need$label %>% recode("0" = 0, "1" = 1) %>% as.numeric(), cex=1)
dev.off()
rm(pred.lg)

data_need <- read.csv(paste(roc_data_path, "roc_ext_set.csv",sep = ""), header=TRUE)
png(file=paste(output_dir, "calibration_ext.png", sep = ""), width=1900, height=1600, res = 300)
val.prob(data_need$pred_f_num_text, data_need$label %>% recode("0" = 0, "1" = 1) %>% as.numeric(), cex=1)
dev.off()
rm(pred.lg)
rm(data_need)


#                 ------------------------- Step 3 DCA analysis-------------------------                   #
#
# -------------------------------------------------------------------------------------------------------- #
initial_path <- paste(project_path, "result\\note_num\\nomogram\\",sep = "")
loop_list <- list("test", "temp", "ext")
for(i in loop_list){
  data_need <- read.csv(paste(initial_path, i, "_dataset.csv",sep = ""), header=TRUE)
  cols <- c(
    "label", "code_status", "code_status_eva_flag", "activity_stand", "vent", 
    "gender", "activity_bed", 
    "activity_sit", "activity_eva_flag", "pao2fio2ratio_vent_flag", 
    "pao2fio2ratio_novent_flag", "bilirubin_max_flag", "alp_max_flag", "alt_max_flag", 
    "ast_max_flag", "baseexcess_min_flag" , "fio2_max_flag", "lactate_max_flag", "lymphocytes_max_flag", 
    "lymphocytes_min_flag", "neutrophils_min_flag", "paco2_max_flag", 
    "pao2_min_flag", "nlr_flag", "vasopressor", "rrt", "gnri_flag"
  )
  data_need[cols] <- lapply(data_need[cols], factor)  ## as.factor() could also be used  
  data_need$label <- as.numeric(data_need$label)-1
  model1 <- decision_curve(label ~ preICU_risk_score + shock_index + code_status + activity_bed + vent + 
                             lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score,
                           data=data_need,
                           #thresholds = seq(0, .4, by = .01),
                           study.design = 'cohort',
                           bootstraps = 500)
  model2 <- decision_curve(label ~ shock_index + code_status + activity_bed + vent + 
                             lactate_max + braden_score + cci_score + resp_rate_mean + gcs_min + temperature_mean,
                           data=data_need,
                           #thresholds = seq(0, .4, by = .01),
                           study.design = 'cohort',
                           bootstraps = 500)
  model3 <- decision_curve(label ~ preICU_risk_score_raw,
                           data=data_need,
                           #thresholds = seq(0, .4, by = .01),
                           study.design = 'cohort',
                           bootstraps = 500)
  model4 <- decision_curve(label ~ saps,
                           data=data_need,
                           #thresholds = seq(0, .4, by = .01),
                           study.design = 'cohort',
                           bootstraps = 500)
  model5 <- decision_curve(label ~ sofa,
                           data=data_need,
                           #thresholds = seq(0, .4, by = .01),
                           study.design = 'cohort',
                           bootstraps = 500)
  png(file=paste(output_dir, "dca_", i, ".png", sep = ""), width=1900, height=1600, res = 300)
  plot_decision_curve( list(model1, model2, model3, model4, model5),
                       curve.names = c("GRACE-ICU", "Structured data score", "preICU risk score", "SAPSII", "SOFA"),
                       col = c("blue", "red", "green", "yellow", "purple"),
                       confidence.intervals = FALSE,  #remove confidence intervals
                       cost.benefit.axis = FALSE, #remove cost benefit axis
                       xlim = c(0, 1), #set xlim
                       legend.position = "topright") #remove the legend
  dev.off()
  rm(model1, model2, model3, model4, model5, data_need)
}
rm(cols, i)


#             ------------------------- Step 4 NRI and IDI compare -------------------------             #
#
# ------------------------------------------------------------------------------------------------------ #
# https://rpubs.com/winterwang/NRIforwuna
# https://blog.csdn.net/JianJuly/article/details/109399017
loop_list <- list("test", "temp", "ext")
for(i in loop_list){
  data_need <- read.csv(paste(output_dir, i, "_all_models_scores_probability.csv",sep = ""), header=TRUE)
  print(paste("--------------------- comparison in ", i, " ---------------------",sep = ""))
  for(j in list("pred_f_num", "preICU_risk_score", "sofa", "saps", "eldericu", "xgb", "rf")){
    print(paste("--------- GRACE-ICU vs.", j, " ----------",sep = ""))
    reclassification(data = data_need, cOutcome = which(names(data_need) == "label"), 
                     predrisk1 = data_need[[j]], 
                     predrisk2 = data_need$pred_f_num_text, 
                     cutoff = c(0, 0.10, 0.30, 1)
    )
    print("Done!")
  }
}

