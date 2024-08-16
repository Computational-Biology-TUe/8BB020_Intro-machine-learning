
# Introduction to machine learning (8BB020)

This course is intended to introduce machine learning principles, explain how commonly use machine learning methods work and illustrating their application to biomedical problems.

## Use of Canvas
This GitHub page contains all the general information about the course and the study materials. The [Canvas page of the course](https://canvas.tue.nl/courses/27567) will be used only for sharing of course information that cannot be made public (e.g. Microsoft Teams links), submission of the practical work and posting questions to the instructors and teaching assistants (in the Discussion section). The students are highly encouraged to use the Discussion section in Canvas. All general questions (e.g. issues with setting up the programming environment, error messages etc., general methodology questions) should be posted in the Discussion section.

**TLDR**: GitHub is for content, Canvas for communication and submission of assignments.

## Schedule

The course schedule is as follows:
* **Lectures**, *time*: Wednesdays 13:30 - 15:30, *location*: Aud. 15 
* **Guided self-study**, *time*: Wednesdays 15.30 - 17.30, *location*: generally in Aud. 07, except for week 1 (Luna 1.050) and week 3 (He. 0.01).

## Practical work

The practical work will be done in groups. The groups will be formed in Canvas and you will also submit all your work there (check the Assignments section for the deadlines). Your are expected to do this work independently with the help of the teaching assistants during the guided self-study sessions (*begeleide zelfstudie*). You can also post your questions in the Discussion section in Canvas at any time (i.e. not just during the practical sessions).

**IMPORTANT: Please read [this guide](how_to_ask_questions.md) on effectively asking questions during the practical sessions.**


### Goal of the practical exercises

The exercises have two goals:

1) Give you the opportunity to obtain 'hands-on' experience in implementing, training and evaluation machine learning models in Python. This experience will also help you better understand the theory covered during the lectures.

2) Occasionally demonstrate some 'exam-style' questions that you can use as a reference when studying for the exam. Note however that the example questions are (as the name suggests) only examples and do not constitute a complete and sufficient list of 'things that you have to learn for the exam'. 

 
### Deliverables

There will be three practical divided over seven weeks as follows:

- Practical 0 (week 1): Introduction to python and python packages for the course
- Practical 1 (week 2-4): Linear models and regularization
- Practical 2 (week 5-7): Neural networks

For Practical 1 and Practical 2 you have to submit deliverables that will be graded and constitute 30% of the final grade. Thus, the work that you do during the practicals has double contribution towards the final grade: as 25% direct contribution and as a preparation for the exam that will define the other 70% of the grade. However note that not all the topics of the exam are covered in the practicals. More details on the specific deliverables are in the notebooks of each practical.

The following rubric will be used when grading the practical work:

| Component  | Insufficient | Satisfactory | Excellent |
| ---------- | ------------- | ------------- | ------------- |
| Code  | Missing or incomplete code structure, runs with errors, lacks documentation  | Self-contained, does not result in errors, contains some documentation, can be easily used to reproduce the reported results | User-friendly, well-structured (good separation of general functionality and experiments, i.e. between .py files and the Pyhthon notebook), detailed documentation, optimized for speed, 
| Answers to questions  | Incorrect, does not convey understanding of the material, appears to be copied from another source | Correct, conveys good understanding of the material, description in own words | Correct, conveys excellent level of understanding, makes connections between topics | 

 

### Use of ChatGPT and other large language models

The use of ChatGPT and other large language models for the practical work is allowed, provided that:

1) You use ChatGPT and other large language models only as aid in your work and not as primary sources of information (e.g. to do literature search), and primary mode of writing and coding (e.g. asking for answers to entire assignment questions is not allowed, however, improving the writing or coding of answers to questions is allowed).
   
2) You write a one-page reflection report on the use of such tools to be submitted along with each assignment answering the following questions:
    * How did you specifically use these tools (give examples)?
    * Were these tools accurate in their answers?
    * What were the up- and down-sides of using ChatGPT (or similar tools) in your work?
    * In your view, are such tools helpful or harmful when used in education?
    * Did it make you more or less productive?

Note that the report is **mandatory** if you used ChatGPT (or similar tools) in any way for the assignment and it does not have any negative consequence (e.g. lead to lower grades). If you do not submit the report we will assume that you did not use such tools but if this is detected during the grading it will be considered cheating. 

# Materials

## Books
The lectures are mainly based on the selected chapters from the following book that is freely available online:

* [An introduction to statistical learning with applications in python](https://www.statlearning.com/), G. James, D. Witten, T. Hastie, R. Tibshirani, J. Taylor

Additional reading materials such as journal articles are listed within the lecture slides.

## Software

**IMPORTANT: It is essential that you correctly set up the Python working environment by the end of the first week of the course so there are no delays in the work on the practicals.**

The practical assignments for this course will be done in Python. Please carefully follow [the instructions available here](software.md) on setting up the working environment and (optionally) a Git workflow for your group.

## Python quiz

**IMPORTANT: Attempting the quiz before the specified deadline is mandatory.**

In the first week of the course you have to do a Python self-assessment quiz in Canvas. The quiz will not be graded. If you fail to complete the quiz before the deadline, you will not get a grade for the course. The goal of the quiz is to give you an idea of the Python programming level that is expected.

If you lack prior knowledge of the Python programming language, you can use the material in the "Python essentials" and "Numerical and scientific computing in Python" modules available [here](https://github.com/tueimage/essential-skills/).

## Lectures and assignments


### Lectures

| # | Date | Title | Slides |
| --- | --- | --- | --- |
| 1 | 04/Sep | Machine learning fundamentals | [intro ](lectures/add), [slides](lectures/add)|
| 2 | 11/Sep | Linear and logistic regression | [slides](lectures/add) |
| 3 | 18/Sep | Regularization for linear models | [slides](lectures/add) | 
| 4 | 25/Sep | Methods for classification | [slides](lectures/add) |
| 5 | 02/Oct | Neural networks, part 1 | [slides](lectures/add) |
| 6 | 09/Oct | Neural networks, part 2 | [slides](lectures/add) |
| 7 | 16/Oct | Unsupervised learning | [slides](lectures/add) |
| 8 | 23/Oct | *No lecture* | - |
| :small_red_triangle:| 31/Oct | *Exam* | |

### Practical assignments

| # | Date | Title | Exercises |
| --- | --- | --- | --- |
| 1 | 04/Sep | Project 0: Numpy refresher | [excercises](practicals/add), [slides](add)|
| 2 | 11/Sep | Project 1.1: Linear and logistic regression | [excercises](practicals/add) |
| 3 | 18/Sep | Project 1.2: Regularization for linear models | [excercises](practicals/add) | 
| 4 | 25/Sep | Project 1.3: Application of linear models to a case study | [excercises](practicals/add) |
| 5 | 02/Oct | Project 2.1: TBD | [excercises](practicals/add) |
| 6 | 09/Oct | Project 2.1: TBD | [excercises](practicals/add) |
| 7 | 16/Oct | Project 2.1: TBD | [excercises](practicals/add) |
| 8 | 23/Oct | *Catch up week!* | - |

Project 0 will not be evaluated. The deadline for submission of Project 1 is in week 5. The deadline for submission of Project 2 is in week 8.

# Other course information

## Learning objectives

After completing the course, the student will be able to:
* define key terminology of machine learning and identify different types of machine learning tasks;
* explain the basic principles behind common machine learning algorithms;
* implement, train and evaluate machine learning models using Python libraries;
* analyze performances of machine learning algorithms and interpret model outputs;
* critically evaluate the strengths and limitations of various machine learning techniques in biomedical data analysis.

## Assessment

The assessment will be performed in the following way:

* Work on the practical assignments: 30% of the final grade (each assignment has equal contribution);

* Final exam: 70% of the final grade.

Intermediate feedback will be provided as grades to the first assignment.

The grading of the assignments will be done per groups, however, it is possible that individual students get separate grade from the rest of the group (e.g. if they did not sufficiently participate in the work of the group).


## Instruction

The students will receive instruction in the following ways:

* Lectures
* Guided practical sessions with the teaching assistants for questions, assistance and advice
* On-line discussion

Course instructors:
* Federica Eduati
* Mitko Veta
* Cian Scannel
* Peter Ebus

Teaching assistants:
* Jesse Quadt
* Myrthe Boone
* Wafae Quorsane
* Mathijs van Gerven
* Simon Habraken

## Recommended prerequisite courses

8BA060 – Linear algebra & multivariate calculus, 8BA050 – Skills experience, 8BA080 – Programming for data analytics.



*This page is carefully filled with all necessary information about the course. When unexpected differences occur between this page and Osiris, the information provided in Osiris is leading.*
