## Step 1. Create the **project** of gemini_icu  
***code***: including all related codes based on PostgreSQL/BigQuery, Python, and R  
***data***:  
1. 'db_generation': saving the data extracted from multiple databases. We have provided example data for 5 episodes of mimiciii, mimiciv, and eicu, respectively.  
2. outlier_range_check.csv: the lower and upper bounds of variables 
3. no_meaning_contents.csv: are considered unuseful or without meaning in the context of the analysis  

***result***: we have provided the optimal preICU risk score if you want to use it directly  
***pre-analysis***: save the ELDER-ICU model (doi: 10.1016/S2589-7500(23)00128-0)
## Step 2. Get the extraction data of mimiciii, mimiciv, and eicu   
## Step 3. Run the code sequentially in the code folder
* Remember check each code file and change the project_path
1. data_process_1.py: obtain the note and tabular data required for the development, temporal, and external validation sets.  
2. older_nlp_mortality_2.ipynb: obtain the optimal pre-ICU risk score and all comparison results with different hyperparameters.
3. older_subtext_nlp_mortality_3.ipynb: obtain the pre-ICU risk score based on the subtext like chief complain.
4. older_feature_select_4.R: select the useful structured variables.
5. older_lr_preicurisk_geminiicu_5.R: acquire the GEMINI-ICU model
6. older_subtext_lr_preicurisk_geminiicu_6.R: acquire the performance of the GEMINI-ICU model (note: using the subtext to acquire the preICU risk score)
7. modeling_eva_comparisons_7.py: compare all of models and scores such as the GEMINI-ICU, preICU risk score, structured data score, ELDER-ICU, XGBoost, RF, SAPSII, and SOFA.
8. models_performance_plot_8.R: plot the needed figures
