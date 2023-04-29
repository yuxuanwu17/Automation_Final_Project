# Prediction of hospital admission at emergency department using passive and active learning


## Background
Emergency department (ED) visit is one of the most common ways to get medical support and ED represents the largest source of hospital admissions. In order to improve the running efficiency of ED, optimize the resource allocation, as well as to maximize the number of patients that get appropriate treatment, we train machine learning models using offline learning, passive and active learning to predict hospital admission at the time of ED triage, using patients' triage information and previous medical history.

**Note**: This is the course project for 02-750 Automation of Scientific Research in Spring 2023


## Data souorce
The Electronic Health Record (EHR) data we use was from a paper published on [PLOS one](https://doi.org/10.1371/journal.pone.0201016). The original retrospective data was obtained from three Emergence Departments from March 2013 to July 2017, each ensuring one year of historical timeframe. We obtained the raw data from this [Kaggle dataset](https://www.kaggle.com/maalona/hospital-triage-and-patient-history-data). 

**Note**: Because data is large (> 500M), which includes 560,486 patient visits with 971 variables and 1 labels, we didn't upload it under this repo. But we can send it to you if your are interested.


## Workflow

<p align="center">
    <img src="https://github.com/yuxuanwu17/Automation_Final_Project/blob/main/figures/workflow.png" height="500" width="700" alt = "workflow"/>
</p>


## File organization

* `feature_processing` folder: includes scripts for feature selection and selected top 100 features.
* `figures` folder: includes all the figures generated in this project.
* `offline` folder: includes scripts of offline learning.
* `active_learning`: includes scripts of active learning.
* `preprocessing` folder: includes scripts for data pre-processing, including removing NA columns and rows, label encoding and so on.
* `data` folder: includes the final data (20,000 records, 100 features) we used for model training and testing.


## Authors
* Yanjing Li, yanjing2@andrew.cmu.edu/liyanjing12138@gmail.com, Carnegie Mellon University
* Sitong Liu , sitongli@andrew.cmu.edu, Carnegie Mellon University
* Xin Wang, xinwang3@andrew.cmu.edu, Carnegie Mellon University
* Yuxuan Wu, yuxuanwu@andrew.cmu.edu, Carnegie Mellon University
