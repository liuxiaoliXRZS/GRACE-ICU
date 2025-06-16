# 0. preprare data - missing ratio
# 1. univariable models
# 2. lasso regression
# 3. multivariable models


#         ----------------------------- Step 0. prepare data ------------------------------            #
#
#         ---------------------------------------------------------------------------------            #
# 0.1 load packages
rm(list = ls())
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
library(epiDisplay) # quickly output OR, 95% CI, P
library(gtsummary) # beautiful three - line table
library("writexl")
project_path <- "...\\grace_icu\\" # change to the project path

paste(project_path, "data\\use\\seq_240\\",sep = "")

# 0.2 set path
num_dir <- paste(project_path, "data\\use\\seq_240\\",sep = "")
text_dir <- paste(project_path, "result\\note\\clinical_longformer\\seq_240\\max_len-512_bs-16_epoch-2_lr-1e-05\\",sep = "")
output_dir <- paste(project_path, "result\\note_num\\",sep = "")
if (dir.exists(output_dir)) {  
  print("The direcoty exists")
} else {  
  # create folder
  dir.create(output_dir)
}

# 0.3 load data and merge note value and numeric values - merge train and val together
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
#     [3] merge dataframes and merge train and validation data
training_dataset <- merge(
  x = subset(text_older[text_older$db_type == 'train',], select = c(id, preICU_risk_score)), 
  y = development_num_older, by = "id")
training_dataset$preICU_risk_score <- round(10*training_dataset$preICU_risk_score)  # more appropriate to expand the data by 10 times
validation_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'val',], select = c(id, preICU_risk_score)), 
  y = development_num_older, by = "id")
validation_dataset$preICU_risk_score <- round(10*validation_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
test_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'test',], select = c(id, preICU_risk_score)), 
  y = development_num_older, by = "id")
test_dataset$preICU_risk_score <- round(10*test_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
temp_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'temp',], select = c(id, preICU_risk_score)), 
  y = temporal_num_older, by = "id")
temp_dataset$preICU_risk_score <- round(10*temp_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
ext_dataset<- merge(
  x = subset(text_older[text_older$db_type == 'ext',], select = c(id, preICU_risk_score)), 
  y = external_num_older, by = "id")  
ext_dataset$preICU_risk_score <- round(10*ext_dataset$preICU_risk_score) # more appropriate to expand the data by 10 times
training_dataset <- rbind(training_dataset, validation_dataset) # combine train and validation set
#      [4] remove no need categorical columns
training_dataset <- subset(training_dataset, select = -c(id,first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
test_dataset <- subset(test_dataset, select = -c(id,first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
temp_dataset <- subset(temp_dataset, select = -c(id,first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
ext_dataset <- subset(ext_dataset, select = -c(id,first_careunit,ethnicity,anchor_year_group, delirium_eva_flag, delirium_flag))
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
ext_dataset[cols] <- lapply(ext_dataset[cols], factor)  ## as.factor() could also be used
rm(development_num_older, temporal_num_older, external_num_older, validation_dataset, text_older, text_dir)


#         ------------------------------ Step 1. univariable ------------------------------            #
#
#         ---------------------------------------------------------------------------------            #
#       [1] define function
Uni_glm_model<- 
  function(x){
    # Fit outcome and variables
    FML<-as.formula(paste0("label==1~",x))
    # glm() logistic regression
    glm1<-glm(FML,data=training_dataset,family = binomial)
    # Extract all regression results into glm2
    glm2<-summary(glm1)
    # 1 - Calculate OR value and round to two decimal places
    OR<-round(exp(coef(glm1)),2)
    # 2 - Extract SE
    SE<-glm2$coefficients[,2]
    # 3 - Calculate CI, round to two decimal places and combine
    CI5<-round(exp(coef(glm1)-1.96*SE),2)
    CI95<-round(exp(coef(glm1)+1.96*SE),2)
    CI<-paste0(CI5,'-',CI95)
    # 4 - Extract P value
    P<-round(glm2$coefficients[,4],2)
    # 5 - Combine variable name, OR, CI, P into a table and remove the first row
    Uni_glm_model <- data.frame('Characteristics'=x,
                                'OR' = OR,
                                'CI' = CI,
                                'P' = P)[-1,]
    # Return the loop function to continue the above operations                    
    return(Uni_glm_model)
  }
variable.names<- colnames(training_dataset)
variable.names <- variable.names[! variable.names %in% c('label', 'sofa', 'oasis', 'saps', 'apsiii')];variable.names

#      [2] Variables are passed into the loop function
Uni_glm <- lapply(variable.names, Uni_glm_model)
# Batch output results and combine them together
Uni_glm<- ldply(Uni_glm,data.frame);Uni_glm

#      [3] The variables that just underwent logistic regression are passed into this glm() function
# Note: This step is only for extracting variables, nothing more
linearvars <- setdiff(colnames(subset(training_dataset, select = -c(sofa,oasis,saps,apsiii))),'label')
linearphrase <- paste(linearvars, collapse=" + ")
fmla <- as.formula( paste0( "label==1~", linearphrase))
names<- glm(fmla,
            data=training_dataset,
            family = binomial)
name<-data.frame(summary(names)$aliased)
# Remove the first row of the row names of the extracted data table and give it to the three - line table
rownames(Uni_glm)<-rownames(name)[-1]
# The original row names of the three - line table are no longer needed, so delete them
Uni_glm <- Uni_glm[,-1]
#       [4] Further modify to the correct names
del_col_Uni <- Uni_glm
del_col_Uni <- del_col_Uni[del_col_Uni$P > 0.05,]
del_col_Uni <- rownames(del_col_Uni)
del_col_Uni <- gsub("1", "", del_col_Uni)
# del_col_Uni <- del_col_Uni[del_col_Uni != "braden_score"] # special add considering the braden score
Uni_glm$P[Uni_glm$P==0] <- "<0.001"
Uni_glm$OR_95CI <- paste0(Uni_glm$OR, ' (', Uni_glm$CI, ')')

if (dir.exists(paste(output_dir, "nomogram/", sep = ""))) {  
  print("The direcoty exists")
} else {  
  # create folder
  dir.create(paste(output_dir, "nomogram/", sep = ""))
}

write.csv(Uni_glm, paste(output_dir,"nomogram/","univariable_models.csv",sep = ""))
rm(name, names, Uni_glm, fmla, linearvars, linearphrase, variable.names)


#         ------------------------------ Step 2. lasso regression ------------------------------            #
#
#         --------------------------------------------------------------------------------------            #
#        [1] Data to matrix function: data.matrix() & remove univariable features not needed
`%ni%` <- Negate(`%in%`)
X <- subset(training_dataset, select = -c(preICU_risk_score,sofa,oasis,saps,apsiii,label))
X <- subset(X,select = names(X) %ni% del_col_Uni)
X <- data.matrix(X)
Y <- training_dataset$label
#        [2]train models and acquire lamda
set.seed(12345)
cv.fit <- cv.glmnet(X,Y,alpha = 1, nfolds =5,
                    family = "binomial",type.measure = "class")
png(filename=paste(output_dir,"nomogram/","lasso_cv_info1.png",sep = ""), width = 1600, height = 1400, res = 300)
plot(cv.fit)
dev.off()
f1 =glmnet(X, Y, family="binomial", alpha=1)
png(filename=paste(output_dir,"nomogram/","lasso_cv_info2.png",sep = ""), width = 1600, height = 1400, res = 300)
plot(f1,xvar="lambda", label=TRUE)
dev.off()
#       [3] Display variables and regression coefficients under two penalty values (tuning coefficients)
lasso.coef1 <- predict(cv.fit, s=cv.fit$lambda.1se, type = "coefficients");lasso.coef1
lasso.coef2 <- predict(cv.fit, s=cv.fit$lambda.min, type = "coefficients");lasso.coef2
#       [4] Obtain model coefficients with fewer parameters
lasso_best <- glmnet(X,Y,alpha = 1,lambda = cv.fit$lambda.1se,
                     family = "binomial")
gx <- list(coef(lasso_best))
coef_info <- data.frame(Group = rownames(gx[[1]]), Value = gx[[1]][,1])
del_col_lassocv <- coef_info
del_col_lassocv <- del_col_lassocv[del_col_lassocv$Value == 0,]
del_col_lassocv <- del_col_lassocv$Group
write.csv(coef_info, paste(output_dir,"nomogram/","lasso_cv_info.csv",sep = ""))
#       [5] Get AIC info - Use forward and backward selection again to remove some features
coef_info <- coef_info[coef_info$Value != 0,]
linearvars <- coef_info$Group[-1]
linearvars <- grep("_flag", linearvars, invert=TRUE, value = TRUE) # remove flag columns
linearphrase <- paste(linearvars, collapse=" + ")
fmla_lasso <- as.formula( paste0("label==1~", linearphrase))
glm1 <- glm(fmla_lasso, family = binomial, data=subset(training_dataset, select = -c(preICU_risk_score,sofa,oasis,saps,apsiii)))
set.seed(12345)
glm2 <- stepAIC(glm1,direction="both")
#       [6]save features name
gx <- list(coef(glm2))
coef_info <- as.data.frame(gx[[1]])
coef_info <- plyr::rename(coef_info, c("gx[[1]]"="coefficient"))
coef_info <- cbind(names = rownames(coef_info), coef_info)
rownames(coef_info) <- 1:nrow(coef_info)
coef_info <- coef_info[-1,]
coef_info$names <- gsub('1', '', coef_info$names)
write.csv(coef_info, paste(output_dir,"nomogram/","stepAIC_info.csv",sep = ""))
use_col_aic <- coef_info$names
rm(glm1, glm2, gx, lasso.coef1, lasso.coef2, lasso_best, fmla_lasso, linearphrase, linearvars, Y, X, f1)



#         ------------------------------ Step 3. multi-variable model ------------------------------            #
#
#         ------------------------------------------------------------------------------------------            #
#linearvars <- use_col_aic
linearvars <- c("gcs_min","cci_score","shock_index","vent","activity_bed","code_status","resp_rate_mean","lactate_max",
                "braden_score","hemoglobin_min","temperature_mean","potassium_max","preICU_risk_score")
linearphrase <- paste(linearvars, collapse=" + ")
fmla_aic <- as.formula( paste0("label==1~", linearphrase))

glm1 <- glm(fmla_aic, family = binomial, data=training_dataset)
# Extract OR, CI, and P values
res <- logistic.display(glm1, simplified=T);res
res <- res$table
res[,'OR'] <- round(res[,'OR'],2)
res[,'lower95ci'] <- round(res[,'lower95ci'],2)
res[,'upper95ci'] <- round(res[,'upper95ci'],2)
res[,'Pr(>|Z|)'] <- round(res[,'Pr(>|Z|)'],5)
res <- as.data.frame(res)
res$OR_95CI <- paste0(res$OR, ' (', res$lower95ci, '-', res$upper95ci, ')')
res$`Pr(>|Z|)`[res$`Pr(>|Z|)`==0] <- "<0.001"
write.csv(res, paste(output_dir,"nomogram/","multivariable_info.csv",sep = ""))
rm(glm1, fmla_aic, linearphrase, linearvars, res)


glm1 <- glm(label ~ preICU_risk_score + shock_index + code_status + activity_bed + vent + 
              lactate_max + cci_score + resp_rate_mean + temperature_mean + gcs_min + braden_score, 
            family = binomial, data=training_dataset)
res <- logistic.display(glm1, simplified=T);res
res <- res$table
write.csv(res, paste(output_dir,"nomogram/","multivariable_info_noround.csv",sep = ""))