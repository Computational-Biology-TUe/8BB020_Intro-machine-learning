
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

## Schedule and course material

Lectures are on Wednesdays 13:30 - 15:30, practicals are on Wednesdays 15.30 - 17.30 (after each lecture). Please check the schedule and canvas announcements: [public link](https://cloud.timeedit.net/nl_tue/web/es/ri1942j60478l0Q19ydj80XlZYQ60k9Q041mjl3tk96lg657060Z68w5ly5q85Zn6QXnZW9rW2Y75d34vzc7RYWQQg20090lX765xWW07jnhWVWVcoxwxjaxjcvXCxxrL6XlWrX8Vavxv7vlvmCrwngycWcXcyAc5Lra9qn7pyWnX5nwcDl6jWvc7W6hrwX0W8chne8523WcjVrc9vXcdvWcc3nwWbbbdmwE82rWcWnW5nEmXnS4Vynzdcw8mj4W9Bt2oX09c529WBDC3S5B92Qo9D46DZC227Ecb9AB90ED77C.phtml) to your schedule. There will be also OGO rooms available if necessary for working in groups during the practical. Here below the schedule for the lectures and the practicals:

### Lectures

⚠️ **Important note for students**:  
Slides marked with **🔴 (2024)** are from last year and are provided only to give you an idea of the course content.  
Slides marked with **🟢 (2025)** are the updated ones for this year and will be uploaded here **before each lecture**.  
The content will remain largely the same, but there may be edits.  
👉 Please **always download the 2025 version of the slides**.

| Week | Date   | Topic                               | Slides |
|------|--------|-------------------------------------|--------|
| 1    | 03/Sep |  Machine learning fundamentals       | [intro](lectures/week1_Intro_2025.pdf) 🟢, [lecture](lectures/week1_lecture_2025.pdf) 🟢 |
| 2    | 10/Sep |  Linear and logistic regression      | [lecture](lectures/week2_lecture_2025.pdf) 🟢 |
| 3    | 17/Sep |  Regularization for linear models    | [lecture](lectures/week3_lecture_2025.pdf) 🟢 |
| 4    | 24/Sep |  SVMs and tree-based methods         | [lecture](lectures/week4_lecture_2025.pdf) 🟢 |
| 5    | 01/Oct |  Neural networks, part 1             | [lecture](lectures/week5_lecture_2024.pdf) 🟢 |
| 6    | 08/Oct |  Neural networks, part 2             | [lecture](lectures/week6_lecture_2025.pdf) 🟢 |
| 7    | 15/Oct |  Unsupervised learning               | [lecture](lectures/week7_lecture_2024.pdf) 🔴 |
| 8 | 22/Oct |  *No lecture* | - |
| :small_red_triangle:| 30/Oct | *Exam* | - |

### Practical assignments

During your first session (Project 0) you are asked to create a Python environment for the course. You can also create your Python environment using the  [`practicals/8BB020_environment.yaml`](practicals/8BB020_environment.yaml) file. [Watch the instruction video on SharePoint — Setting up the Python environment](https://tuenl-my.sharepoint.com/:v:/g/personal/f_eduati_tue_nl/ETB31YZYvKFHuXFgixBO-IQBgkQqfuxd09VXFQvYxeNSEA?e=zv20Oa)


| # | Date |  Title | Exercises |
| --- |  --- | --- | --- |
| 1 | 03/Sep | Project 0: Introduction | [project 0](practicals/part0_intro.ipynb) |
| 2 | 10/Sep | Project 1.1: Linear and logistic regression | [project 1](practicals/part_1_linear_models.ipynb) |
| 3 | 17/Sep | Project 1.2: Regularization for linear models |  | 
| 4 | 24/Sep | Project 1.3: Application of linear models to a case study |  |
| 5 | 01/Oct | Project 2.1: Neural networks, part 1 | [project 2](practicals/part_2.ipynb) |
| 6 | 08/Oct | Project 2.1: Neural networks, part 2 |  |
| 7 | 15/Oct | Project 2.1: Application of neural networks to a case study |  |
| 8 | 22/Oct | *No practical* | - |

### Other course material
The lectures are mainly based on the selected chapters from the following book that is freely available online:

* [An introduction to statistical learning with applications in python](https://www.statlearning.com/), G. James, D. Witten, T. Hastie, R. Tibshirani, J. Taylor

Additional reading materials such as journal articles are listed within the lecture slides.


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
| **Code** | Missing or incomplete; does not run; no documentation | Runs without errors; notebook can be executed from start to finish without manual changes; some comments/markdown; results reproducible directly | Runs cleanly from top to bottom; clear structure with separation of reusable code in `.py` files and experiments in notebooks; consistent style; well-documented with meaningful comments and markdown; reasonably optimized |
| **Answers** | Incorrect or missing; shows little understanding; appears to be copied from elsewhere | Mostly correct; shows good understanding; written in own words | Correct and precise; demonstrates deep understanding; connects multiple topics; justifies design choices |

### How to succeed in the practicals
> 💡 Tips to get the most out of the practicals:
- **Work steadily:** Don’t leave practicals until the last minute — they are designed to reinforce the lectures.  
- **Engage in discussion:** Ask questions in Canvas Discussions; if you have a question, chances are others do too.  
- **Connect practice to theory:** Use the hands-on exercises to check if you really understand how the algorithms work.  
- **Reflect on choices:** When writing answers, explain not only *what* you did but also *why* — this is key to demonstrating deep understanding.  

## Exam

The written exam is designed to **assess your understanding** of the topics covered in the course.  
It covers **all lecture material**, unless the instructor explicitly states otherwise.  
Do not assume that only practical content or example questions reflect the scope of the exam.  

The exam will consist of a **combination of open-ended questions and multiple-choice questions**.  
- **No programming:** There will be no programming in the written exam. The programming component is evaluated through the practical projects.  
- **Examples of questions:**  
  - Multiple-choice questions will be similar in style to the Mentimeter questions asked during the lectures.  
  - Open-ended questions will be similar in style to the conceptual (red) questions in the practical assignments, excluding those that involve or reference coding.

These are examples only and do not represent the complete set of possible exam questions.  

### Preparing for the exam
Good preparation combines lectures, practicals, and self-study:
- **Lectures:** Actively attend and engage with the material. Use lectures to build your first understanding of each topic, and take the opportunity to **ask questions** if something is unclear. Slides are provided, and you are expected to consult the corresponding textbook chapters.  
- **Practicals:** Actively participate in the group-based, hands-on projects. They help you consolidate selected methods in practice, and you should make use of TA support and Canvas Discussions to resolve questions. Some *exam-style* questions are included, but these are only examples and **do not represent the full scope** of the exam. The exam covers all lecture topics, while the practicals focus on a subset.  
- **Self-study:** A very important part of your preparation. Spend sufficient time reviewing the lecture material independently — both slides and textbook chapters — and make sure you can explain the concepts and methods in your own words. Self-study is where you move from familiarity to real understanding.  

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


# Other course information

## Learning objectives

After completing the course, the student will be able to:
* define key terminology of machine learning and identify different types of machine learning tasks;
* explain the basic principles behind common machine learning algorithms;
* implement, train and evaluate machine learning models using Python libraries;
* analyze performances of machine learning algorithms and interpret model outputs;
* critically evaluate the strengths and limitations of various machine learning techniques in biomedical data analysis.

## Assessment

Final grade composition:  
- **Written exam:** 70%  
- **Practical work (Projects 1 & 2):** 30%  
  - Each project contributes **equally (15%)** to the final grade.  

Intermediate feedback will be provided as grades to the first assignment.

The grading of the assignments will be done per groups, however, it is possible that individual students get separate grade from the rest of the group (e.g. if they did not sufficiently participate in the work of the group).

## Course workload and study expectation
This is a 5 ECTS course, equivalent to 140 study hours. Spread over 9 weeks (with week 9 being the exam week), this averages to about 15.5 hours per week.  
- Contact hours: 2-hour lecture + 2-hour practical each week.  
- Independent study: ~11–12 hours per week are expected for reviewing lecture material and self-study.  
These are general guidelines; individual needs may vary. Concepts will become clearer through consistent self-study, and if you encounter difficulties you are encouraged to ask questions during lectures, practicals, or on Canvas.  

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
