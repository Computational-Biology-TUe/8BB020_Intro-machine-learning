
# Introduction to machine learning (8BB020)

This course is intended to introduce machine learning principles, explain how commonly use machine learning methods work and illustrating their application to biomedical problems.

## Use of Canvas
This GitHub page contains all the general information about the course and the study materials. The [Canvas page of the course](https://canvas.tue.nl/courses/27567) will be used only for sharing of course information that cannot be made public (e.g. Microsoft Teams links), submission of the practical work and posting questions to the instructors and teaching assistants (in the Discussion section). The students are highly encouraged to use the Discussion section in Canvas. All general questions (e.g. issues with setting up the programming environment, error messages etc., general methodology questions) should be posted in the Discussion section.

**TLDR**: GitHub is for content, Canvas for communication and submission of assignments.

## Lectures and practicals

Lectures are on Wednesdays 13:30 - 15:30, practicals are on Wednesdays 15.30 - 17.30 (after each lecture). Here below the schedule for the lectures and the practicals:

### Lectures

⚠️ **Important note for students**:  
Slides marked with **🔴 (2024)** are from last year and are provided only to give you an idea of the course content.  
Slides marked with **🟢 (2025)** are the updated ones for this year and will be uploaded here **before each lecture**.  
The content will remain largely the same, but there may be edits.  
👉 Please **always download the 2025 version of the slides**.

| Week | Date   | Room   | Topic                               | Slides |
|------|--------|--------|-------------------------------------|--------|
| 1    | 03/Sep |  | Machine learning fundamentals       | [intro](lectures/week1_Intro_2024.pdf) 🔴, [lecture](lectures/week1_lecture_2024.pdf) 🔴 |
| 2    | 10/Sep |  | Linear and logistic regression      | [lecture](lectures/week2_lecture_2024.pdf) 🔴 |
| 3    | 17/Sep |  | Regularization for linear models    | [lecture](lectures/week3_lecture_2024.pdf) 🔴 |
| 4    | 24/Sep |  | SVMs and tree-based methods         | [lecture](lectures/week4_lecture_2024.pdf) 🔴 |
| 5    | 01/Oct |  | Neural networks, part 1             | [lecture](lectures/week5_lecture_2024.pdf) 🔴 |
| 6    | 08/Oct |  | Neural networks, part 2             | [lecture](lectures/week6_lecture_2024.pdf) 🔴 |
| 7    | 15/Oct |  | Unsupervised learning               | [lecture](lectures/week7_lecture_2024.pdf) 🔴 |
| 8 | 22/Oct |  | *No lecture* | - |
| :small_red_triangle:| 30/Oct | *Exam* | - |

### Practical assignments

| # | Date | Location | Title | Exercises |
| --- | --- | --- | --- | --- |
| 1 | 03/Sep |  | Project 0: Introduction |  |
| 2 | 10/Sep |  | Project 1.1: Linear and logistic regression | |
| 3 | 17/Sep |  | Project 1.2: Regularization for linear models |  | 
| 4 | 24/Sep |  | Project 1.3: Application of linear models to a case study |  |
| 5 | 01/Oct |  | Project 2.1: Neural networks, part 1 | |
| 6 | 08/Oct |  | Project 2.1: Neural networks, part 2 |  |
| 7 | 15/Oct |  | Project 2.1: Application of neural networks to a case study |  |
| 8 | 22/Oct | - | *No practical* | - |

<ins>*Please note that the room all upcoming practical sessions has been changed, the new room is Gem-Z 3A.05</ins>

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

 
## Use of ChatGPT, GitHub Copilot, and other AI tools

The use of ChatGPT, GitHub Copilot, and similar AI tools is **allowed** for the practical work, provided that:  

- They are used as **support** (e.g. improving writing or debugging code), not as the **primary source** of information or a way to generate full assignment answers.  
- If you use such tools, you must submit a **one-page reflection report** with your assignment, addressing:  
  1. How you used the tool (with examples)  
  2. Whether the answers were accurate  
  3. Pros and cons of using it in your work  
  4. Whether you found it **useful or harmful** in education  
  5. Its impact on your productivity  

The reflection report is **mandatory** if you used AI tools. Not submitting it while having used them will be considered **cheating**.  

According to the official [TU/e Working Agreement for AI Use in Education (PDF)](https://assets.w3.tue.nl/w/fileadmin/Education_Guide/Content/Programs/Testing%20and%20assessment/AI%20Rules_TUe.pdf):  

-  **Allowed uses**: brainstorming, summarizing, refining writing, translating, and using AI as a sparring partner — unless explicitly forbidden by the examiner.  
-  **AI is not a reliable source**: outputs must always be critically evaluated.  
-  **Transparency required**: if AI replaces or generates part of your work, you must name the tool, version, and describe how you used it.  
-  **Not allowed**: generating research data (quantitative or qualitative) with AI, unless explicitly permitted.  
-  **Fraud risk**: undeclared or disallowed use of AI can be considered fraud. AI-detection tools alone are *not* sufficient proof, but checks (e.g. oral exams) may be applied.  

Further information:  
- [Tips & lessons learned from previous students](./AI_tips.md) 

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

Teaching assistants:
* Simon Habraken
* Bram Hormann
* Niels van Noort
* Jelle van der Pas

## Recommended prerequisite courses

8BA060 – Linear algebra & multivariate calculus, 8BA050 – Skills experience, 8BA080 – Programming for data analytics.



*This page is carefully filled with all necessary information about the course. When unexpected differences occur between this page and Osiris, the information provided in Osiris is leading.*
