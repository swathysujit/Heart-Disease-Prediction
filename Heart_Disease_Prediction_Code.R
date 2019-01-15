# Importing Heart Disease Training data set
#install.packages("caret")
#install.packages("caTools")
library(dplyr)
library(caret)
library(caTools)
library(ggplot2)

#Reading the training and test sets csv files
dataset <- read.csv("train_values.csv")
test_set <- read.csv("test_values.csv")

#Visualizing before scaling continuouos variables
#Checking if the continuous variables have effect on the heart_disease_present factor
dataset$heart_disease_present = as.factor(dataset$heart_disease_present)

#resting_blood_pressure does not seem to affect, as medians are in the same line
boxplot(dataset$resting_blood_pressure~dataset$heart_disease_present)

#There is some effect of serum cholestrol, oldpeak,age and max heart rate
boxplot(dataset$serum_cholesterol_mg_per_dl~dataset$heart_disease_present)
boxplot(dataset$oldpeak_eq_st_depression~dataset$heart_disease_present)
boxplot(dataset$age~dataset$heart_disease_present)
boxplot(dataset$max_heart_rate_achieved~dataset$heart_disease_present)


#Storing the patient ids of test set for future reference and for final submission along with results
patient_id <- test_set$patient_id

#Removing the patient id column from training and test set
dataset <- dataset %>% select(-patient_id) 
test_set <- test_set %>% select(-patient_id)

#Checking for missing values and treatment of missing values in dataset and test set
sapply(dataset,function(x) sum(is.na(x)))
sapply(test_set,function(x) sum(is.na(x)))

#Checking distribution of values for each parameter for dataset and test_set
sapply(dataset,summary)
sapply(test_set,summary)

#Converting categorical columns into 'factor' type, for dataset and test_set
#using dplyr, better way to convert to factor all together than the other way(done for test_set below)
dataset <- dataset %>% mutate_at(vars(heart_disease_present,thal,sex,chest_pain_type,
                                      exercise_induced_angina,num_major_vessels,
                                      slope_of_peak_exercise_st_segment, resting_ekg_results,
                                      fasting_blood_sugar_gt_120_mg_per_dl), factor)


#heart_disease_present column is not there in the test_set
test_set$thal=as.factor(test_set$thal)
test_set$chest_pain_type=as.factor(test_set$chest_pain_type)
test_set$exercise_induced_angina=as.factor(test_set$exercise_induced_angina)
test_set$sex=as.factor(test_set$sex)
test_set$slope_of_peak_exercise_st_segment=as.factor(test_set$slope_of_peak_exercise_st_segment)
test_set$num_major_vessels=as.factor(test_set$num_major_vessels)
test_set$fasting_blood_sugar_gt_120_mg_per_dl=as.factor(test_set$fasting_blood_sugar_gt_120_mg_per_dl)
test_set$resting_ekg_results=as.factor(test_set$resting_ekg_results)

#Checking class of dataset and test_set to ensure data types and factors are assigned correctly
sapply(dataset,class)
sapply(test_set,class)

#Checking levels of the factors
levels(dataset$thal)

#Scaling the continuous variables/columns before splitting into training and validation
#so that the scaling happens independently for both sets and not based on all values
#Performing scaling for both dataset and test_set
dataset[c(3,8,9,11,12)]<-scale(dataset[c(3,8,9,11,12)])

test_set[c(3,8,9,11,12)]<-scale(test_set[c(3,8,9,11,12)])

#Checking class/data types of each column of dataset and test_set to be sure of the changes
sapply(dataset,class)
sapply(test_set,class)

#Visualising Factor-Factor relationships
dataset %>% ggplot(aes(x=heart_disease_present))+
 geom_bar(aes(fill = slope_of_peak_exercise_st_segment),position = "fill")
  
dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = thal),position = "fill")

dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = chest_pain_type),position = "fill")

dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = num_major_vessels),position = "fill")

#'Fasting_blood_sugar doesn't seem to effect hear_disease_present, equal split as seen in plot
dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = fasting_blood_sugar_gt_120_mg_per_dl),position = "fill")

dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = resting_ekg_results),position = "fill")

dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = sex),position = "fill")

dataset %>% ggplot(aes(x=heart_disease_present))+
  geom_bar(aes(fill = exercise_induced_angina),position = "fill")

#Splitting into the training data- 'dataset' further into training and validation data sets
set.seed(123)
validation_split <- sample.split(dataset$heart_disease_present,SplitRatio = 0.75)
training_set <- subset(dataset,validation_split==TRUE)
validation_set <- subset(dataset,validation_split==FALSE)
  
#Checking even split of '0' and '1' in training and validation set and the original dataset
mean(as.numeric(training_set$heart_disease_present)-1)
mean(as.numeric(validation_set$heart_disease_present)-1)
mean(as.numeric(dataset$heart_disease_present)-1)

#Exploratory Data Analysis
dim(training_set)
sapply(training_set,class)
sapply(training_set,mean)
head(training_set)
levels(training_set$heart_disease_present)
table(training_set$heart_disease_present)
table(validation_set$heart_disease_present)

summary(training_set)
#Finding impact of each parameter and the related statistics to decide input parameters for the classifier using
#linear regression model (similar to anova test)
lm_full <- glm(formula = heart_disease_present~.,data=training_set,family =binomial)
lm_null <- glm(formula = heart_disease_present~1,data=training_set,family =binomial)
lm_optimal <- step(lm_null, scope = list(upper = lm_full), direction = "forward")
lm_optimal2 <- step(lm_null, scope = list(upper = lm_full), direction = "both")
lm_optimal3 <- step(lm_full, scope = list(lower = lm_null), direction = "backward")
summary(lm_optimal)
summary(lm_optimal2)
#Visualizing the relationship of dependent and independents to better understand 
x <- training_set[,1:13]
y <- training_set[,14]
par(mfrow = c(1,13))
for(i in 1:13) {
  boxplot(x[,i], main = names(training_set)[i])
}

boxplot(training_set$heart_disease_present,training_set$resting_blood_pressure)

#Based on glm model, and AIC forward step feature selection method, which is used
#for model performance check for classification model which is the case here, we can
#only consider columns - thal,num_major_vessels,exercise_induced,angina,
#chest_pain,Oldpeak_eq_st_depression,sex and resting_ekg_results in our classification model
#We can now build classification model based on these 7 independents and check accuracy
#We can reiterate and add/remove and again compute accuracy later


#Building a Naive Bayes model for classification  
install.packages("e1071")
library(e1071)

classifier <- naiveBayes(formula = heart_disease_present~thal+chest_pain_type+num_major_vessels+
                           resting_ekg_results+oldpeak_eq_st_depression+sex+exercise_induced_angina,
                         data = training_set)

validation_results<-predict(classifier,validation_set[,c(2,4,5,7,9,10,13)],type="raw")
validation_results[,2]
cm = table(validation_results,validation_set$heart_disease_present)


test_results<-predict(classifier,test_set[,c(2,4,5,7,9,10,13)],type="raw")
test_final <- cbind(patient_id,test_results[,2])
write.csv(test_final,"test_final.csv",row.names = FALSE)
