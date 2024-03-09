# AII-AI_Assisted_Identification_of_Disadvantaged_Students
Delivered as a project for the course "Artificial Intelligence in Industry"


-------
This project is under the topic of “Study and application of fairness techniques”, it focuses on the use cases actively investigated in the Aequitas project. From the various use cases, here my discussion mainly is about the USE-CASE-S2.

This project is focus on the use cases actively investigated in the Aequitas project. It consists of three tasks:

1. training a model to predict the current academic performance of the students (dependent variable) in each grade and academic year using the *student, principals, family and teachers information *(independent variables)**
2. Fairness bias
3. Missing data bias

The dataset is a part of the aequitas consortium, it contains information on the educational system in Canary islands. It is composed of **census** (all student population) and **sample** data on compulsory primary (third and sixth grade) and secondary (fourth grade) education students enrolled in 2015-2016, 2016-2017, 2017-2018 and 2018-2019 academic years.

The problem is almost each variable in the database has missing data. I implemented several imputation methods, like: simple_most_frequent, knn, Matrix Completion and Deep Generative Modelling method. And comparing simple_most_frequent and knn methods don't change much in terms of predicting results, so I use the simple_most_frequent method for the fairness test later on.

Based on my experiments, I was able to make the dataset essentially average in distribution for each sensitive attribute, and for DIDI, essentially reduce the bias by a factor of 3 of the dataset's orignal DIDI value.
