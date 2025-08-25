
# Introduction to machine learning (8BB020)

Machine learning is becoming increasingly important in biomedical engineering, as hospitals and research labs generate vast and growing amounts of data. These methods are powerful for identifying patterns in such complex datasets, but to apply them properly it is essential to understand how the underlying algorithms work. This course provides a foundation in machine learning, focusing on how models are trained from data and the principles behind different algorithms. Practical sessions are included to reinforce the theory through hands-on application.

## Use of Canvas

> **GitHub = course content (slides, notebooks, instructions)**  
> **Canvas = announcements, submissions, and discussion**

This GitHub repository contains the course information and study materials.  
The [Canvas page](https://canvas.tue.nl/courses/31400) is used only for:  
- Announcements and course information that cannot be made public (e.g. possible links to video recordings)  
- Submission of practical work  
- Posting questions in the **Discussion** section  

Students are strongly encouraged to use the Canvas Discussion section for all general questions (e.g. programming environment setup, error messages, or methodology). Chances are that other students have the same question, and posting it there allows everyone to benefit and turn it into an interactive discussion. 

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

Practical sessions are designed to reinforce the lectures and help you consolidate the theory through hands-on exercises.  

### Goals of the practicals
1. Gain experience in implementing, training, and evaluating machine learning models in Python. This practice supports a deeper understanding of the theory.  
2. Encounter occasional *exam-style* questions that serve as reference when preparing for the exam. (These are only examples — not a complete list of what may appear on the exam.)  

### Group work
- Practical work is carried out in **small groups**.  
- Groups will be **formed via Canvas** at the start of the course.  
- Each group submits one set of deliverables.  

### Deliverables and grading
- **Projects:**  
  - Project 1 (Weeks 2–4)  
  - Project 2 (Weeks 5–7)  
- **Submission:** via Canvas (see practical notebooks for details).  
- **Weight:** Practical work counts for **30% of the final grade**. The remaining 70% comes from the written exam.  
- **Double benefit:** Practical work contributes directly to your grade *and* prepares you for the exam. Note that not all exam topics are covered in the practicals.  

### Support during practicals

Teaching assistants (TAs) will be available during practical sessions to clarify instructions and help you debug issues; they will not provide full solutions.  
For questions outside practical hours, please use **Canvas Discussions**. Please read [this guide](how_to_ask_questions.md) on effectively asking questions during the practical sessions.

### Grading rubric
Your practical work will be evaluated on two components: **code** and **answers to questions**.


| Component | Insufficient | Satisfactory | Excellent |
|-----------|--------------|--------------|-----------|
| **Code** | - Code is missing or incomplete  
- Does not run without errors  
- No or minimal comments/documentation | - All required code is present  
- Runs **without modification** from start to finish in the notebook (no hidden steps or manual fixes needed)  
- Functions are reasonably named and there is some documentation (comments or markdown)  
- Results in the notebook can be reproduced directly by running all cells in order | - Code runs cleanly from top to bottom in one go (no manual edits or reordering)  
- Clear structure with separation of reusable code in `.py` files and experiments in notebooks  
- Consistent style (naming, formatting)  
- Well-documented with meaningful comments and markdown explaining the steps  
- Efficient and reasonably optimized where possible |
| **Answers** | - Incorrect or missing  
- Shows little or no understanding  
- Copy-pasted from external source | - Mostly correct  
- Shows understanding of the material  
- Written in own words | - Correct and precise  
- Shows deep understanding of the concepts  
- Makes explicit connections between topics  
- Provides reasoning for choices made |

| Component  | Insufficient | Satisfactory | Excellent |
| ---------- | ------------- | ------------- | ------------- |
| Code  | Missing or incomplete code structure, runs with errors, lacks documentation  | Self-contained, does not result in errors, contains some documentation, can be easily used to reproduce the reported results | User-friendly, well-structured (good separation of general functionality and experiments, i.e. between .py files and the Pyhthon notebook), detailed documentation, optimized for speed, 
| Answers to questions  | Incorrect, does not convey understanding of the material, appears to be copied from another source | Correct, conveys good understanding of the material, description in own words | Correct, conveys excellent level of understanding, makes connections between topics | 

### How to succeed in the practicals
> 💡 Tips to get the most out of the practicals:
- **Work steadily:** Don’t leave practicals until the last minute — they are designed to reinforce the lectures.  
- **Engage in discussion:** Ask questions in Canvas Discussions; if you have a question, chances are others do too.  
- **Connect practice to theory:** Use the hands-on exercises to check if you really understand how the algorithms work.  
- **Reflect on choices:** When writing answers, explain not only *what* you did but also *why* — this is key to demonstrating deep understanding.  

 
## Use of ChatGPT, GitHub Copilot, and other AI tools

The use of generative AI assistants such as **ChatGPT, GitHub Copilot, and other large language model (LLM)-based tools** is **allowed** in this course, provided that you follow the official [TU/e Working Agreement for AI Use in Education (Nov 2024, PDF)](https://assets.w3.tue.nl/w/fileadmin/Education_Guide/Content/Programs/Testing%20and%20assessment/AI%20Rules_TUe.pdf).

### Key points from the TU/e rules
- **Allowed uses**: brainstorming, summarizing, refining writing, translating, and using AI as a sparring partner — unless explicitly forbidden by the examiner.  
- **AI is not a reliable source**: outputs must always be critically evaluated.  
- **Transparency required**: if AI replaces or generates part of your work, you must name the tool, version, and describe how you used it.  
- **Not allowed**: generating research data (quantitative or qualitative) with AI, unless explicitly permitted. 
- **Fraud risk**: undeclared or disallowed use of AI can be considered fraud and checks (e.g. oral exams) may be applied.

### Course-specific requirement
ChatGPT, GitHub Copilot, or similar LLM-based tools can be used as **support** (e.g. improving writing or debugging code), not as the **primary source** of information or a way to generate full assignment answers. 
If you use such tools in your assignments, you must submit a **one-page reflection report** together with your work.  
This report should address:  
1. How you used the tool (with examples)  
2. Whether the answers were accurate  
3. Pros and cons of using it in your work  
4. Whether you found it **useful or harmful** in education  
5. Its impact on your productivity  

The reflection report is **mandatory** if you used AI tools. Not submitting it while having used them will be considered **cheating**.  

Further information: [Tips & lessons learned from previous students](./AI_tips.md)  

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
