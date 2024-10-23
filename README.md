**AI-Based Student Assessment System**

**Overview**

The AI-Based Student Assessment System is a machine learning-powered application designed to assess student performance and predict potential challenges students may face during placement drives. The system evaluates student readiness for remedial classes and provides insights into academic performance by analyzing subject marks, attendance, and interest in classes.

This system consists of two primary sections:

**Student Model: **Determines if a student is eligible for remedial classes based on their internal marks.
**Teacher Model:** Provides teachers with insights into class interest and student readiness for placements based on average attendance and performance.
Features
Student Assessment: Predicts whether students need to take remedial classes based on their subject marks.
Teacher Insights: Calculates class interest levels based on average attendance and subject results.
Subject Mark Evaluation: Provides feedback on individual subjects using a rating system from 0 to 5, based on percentage ranges.
Streamlit Interface: A user-friendly interface where students can input their marks, and teachers can view student readiness evaluations.
Key Functionality
**Student Section**:
Students can input marks for individual subjects.
The system will evaluate the input and provide feedback on remedial class eligibility.
It uses machine learning to classify students based on predefined performance thresholds.
**Teacher Section:**
Teachers can view aggregated data, such as average class interest, based on student attendance and marks.
Provides ratings from 1 to 5 for each subject and the overall class interest.
Session Management: The system stores and transfers student input from the student section to the teacher section using session state.
