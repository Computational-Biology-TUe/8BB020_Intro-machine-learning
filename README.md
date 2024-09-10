
# Introduction to machine learning (8BB020)

This course is intended to introduce machine learning principles, explain how commonly use machine learning methods work and illustrating their application to biomedical problems.

## Use of Canvas
This GitHub page contains all the general information about the course and the study materials. The [Canvas page of the course](https://canvas.tue.nl/courses/27567) will be used only for sharing of course information that cannot be made public (e.g. Microsoft Teams links), submission of the practical work and posting questions to the instructors and teaching assistants (in the Discussion section). The students are highly encouraged to use the Discussion section in Canvas. All general questions (e.g. issues with setting up the programming environment, error messages etc., general methodology questions) should be posted in the Discussion section.

**TLDR**: GitHub is for content, Canvas for communication and submission of assignments.

## Lectures and practicals

Lectures are on Wednesdays 13:30 - 15:30, practicals are on Wednesdays 15.30 - 17.30 (after each lecture). Here below the schedule for the lectures and the practicals:

### Lectures

| # | Date | Location | Title | Slides |
| --- | --- | --- | --- | --- |
| 1 | 04/Sep | Aud.15 | Machine learning fundamentals | [intro](lectures/week1_Intro.pdf), [lecture](lectures/week1_lecture.pdf) |
| 2 | 11/Sep | Aud.15 | Linear and logistic regression | [lecture](lectures/week1_lecture.pdf) |
| 3 | 18/Sep | Aud.15 | Regularization for linear models |  | 
| 4 | 25/Sep | Aud.15 | Methods for classification |  |
| 5 | 02/Oct | Aud.15 | Neural networks, part 1 |  |
| 6 | 09/Oct | Aud.15 | Neural networks, part 2 |  |
| 7 | 16/Oct | Aud.15 | Unsupervised learning |  |
| 8 | 23/Oct | - | *No lecture* | - |
| :small_red_triangle:| 31/Oct | *Exam* | |

### Practical assignments

| # | Date | Location | Title | Exercises |
| --- | --- | --- | --- | --- |
| 1 | 04/Sep | ~~ ~~Luna 1.050~~ ~~ Gem-Z 3A.05 * | Project 0: Introduction | [project 0](practicals/part0_intro.ipynb) |
| 2 | 11/Sep | Aud. 07 | Project 1.1: Linear and logistic regression | [project 1](practicals/part_1_linear_models.ipynb) |
| 3 | 18/Sep | He. 0.01 | Project 1.2: Regularization for linear models |  | 
| 4 | 25/Sep | Aud. 07 | Project 1.3: Application of linear models to a case study |  |
| 5 | 02/Oct | Aud. 07 | Project 2.1: Neural networks, part 1 |  |
| 6 | 09/Oct | Aud. 07 | Project 2.1: Neural networks, part 2 |  |
| 7 | 16/Oct | Aud. 07 | Project 2.1: Application of neural networks to a case study |  |
| 8 | 23/Oct | - | *No practical* | - |

*Please note that the room for the first practical has been changed, the new room is Gem-Z 3A.05

## Practical work

The practical work will be done in groups. The groups will be formed in Canvas and you will also submit all your work there (check the Assignments section for the deadlines). Your are expected to do this work independently with the help of the teaching assistants during the guided self-study sessions (*begeleide zelfstudie*). You can also post your questions in the Discussion section in Canvas at any time (i.e. not just during the practical sessions).

**IMPORTANT: Please read [this guide](how_to_ask_questions.md) on effectively asking questions during the practical sessions.**


### Goal of the practical exercises

The exercises have two goals:

1) Give you the opportunity to obtain 'hands-on' experience in implementing, training and evaluation machine learning models in Python. This experience will also help you better understand the theory covered during the lectures.

2) Occasionally demonstrate some 'exam-style' questions that you can use as a reference when studying for the exam. Note however that the example questions are (as the name suggests) only examples and do not constitute a complete and sufficient list of 'things that you have to learn for the exam'. 

 
### Deliverables

For Project 1 (weeks 2-4) and Project 2 (week 5-7) you have to submit deliverables that will be graded and constitute 30% of the final grade. Thus, the work that you do during the practicals has double contribution towards the final grade: as 30% direct contribution and as a preparation for the exam that will define the other 70% of the grade. However note that not all the topics of the exam are covered in the practicals. More details on the specific deliverables are in the notebooks of each practical.

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
