import pandas as pd
import numpy as np
import random
import streamlit as st
from fpdf import FPDF  # For generating PDF files
import matplotlib.pyplot as plt  # For plotting charts
import pickle  # For saving the trained model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

st.markdown("""
        <div style='
            font-family: Arial, sans-serif; 
            color: DarkBlue; 
            background-color: #f0f8ff; 
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            border: 2px solid #4682B4;
            width: 100%;
            margin: auto;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            '>
            <h2 style='font-size: 24px; margin: 0;'>Artificial Intelligence Based Student Assessment System</h2>
        </div>
        """, unsafe_allow_html=True)

st.info("An Application where the student can assess himself/herself or a teacher can assess the student.\n**[NOTE:-Exclusive for 6th SEM]**")


#Student Model
def load_preprocess_student_data(filepath):
    ## load the data into DataFrame
    studs_data1= pd.read_csv(filepath)

    # Drop the 'student_id' column if it exists
    if 'student_id' in studs_data1.columns:
        studs_data1 = studs_data1.drop(columns=['student_id'])

    # Select only numeric columns for calculating the median
    numeric_columns = studs_data1.select_dtypes(include='number').columns

    # Handle missing values by filling with the median for numeric columns only
    studs_data1[numeric_columns] = studs_data1[numeric_columns].fillna(studs_data1[numeric_columns].median())

    # If you want to fill non-numeric columns with a specific value (e.g., a string), you can do that separately
    # For example, filling non-numeric columns with 'Unknown'
    non_numeric_columns = studs_data1.select_dtypes(exclude='number').columns
    studs_data1[non_numeric_columns] = studs_data1[non_numeric_columns].fillna('Unknown')

    # Separate features (X) and target (y)
    X = studs_data1.drop(columns=['placement_readiness'])
    y = studs_data1['placement_readiness']
    
    return X, y

def train_student_model(X_train,y_train):
    
    # Standardize the feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the model
    model.fit(X_train_scaled, y_train)
    return model,scaler
    #X_test_scaled = scaler.transform(X_test)
    # Make predictions
    #y_pred = model.predict(X_test_scaled)

def load_preprocess_teacher_data(filepath):   
    # Load the data into a DataFrame
    teach_data1 = pd.read_csv(filepath)
    # Drop rows with missing values
    teach_data1 = teach_data1.dropna()
    # Drop any columns not used in the model
    teach_data1= teach_data1.drop(columns=['student_id'], errors='ignore')
    # Features and target variable
    X = teach_data1[['avg_attendance', 'avg_score']]
    y = teach_data1['avg_class_interest']
    return X, y
def train_teacher_model(X_train,y_train):
    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Training the model
    model_teacher = RandomForestRegressor(n_estimators=100, random_state=42)
    model_teacher.fit(X_train_scaled, y_train)
    return model_teacher, scaler
def create_studentpdf(name,student_id,new_student, prediction, remedial_subjects,pe2_subject,oe2_subject,oe3_subject):
    pdf = FPDF()
    pdf.add_page()
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Student Assessment Report', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Name: "f'{name}', 0, 1)
    pdf.cell(0, 10, "USN: "f'{student_id}', 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Student Scores:', 0, 1)

    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Computer Networks: {new_student["computer_nw_score"]}', 0, 1)
    pdf.cell(0, 10, f'Software Engineering: {new_student["software_engg_score"]}', 0, 1)
    pdf.cell(0, 10, f'Theoretical Foundations Of Computer Science (TFCS): {new_student["tfcs_score"]}', 0, 1)
    pdf.cell(0, 10, f'{pe2_subject}: {new_student["pe2_score"]}', 0, 1)
    pdf.cell(0, 10, f'{oe2_subject}: {new_student["oe2_score"]}', 0, 1)
    pdf.cell(0, 10, f'{oe3_subject}: {new_student["oe3_score"]}', 0, 1)
    pdf.cell(0, 10, f'Indian Knowledge System (IKS): {new_student["iks_score"]}', 0, 1)
    pdf.cell(0, 10, f'CGPA: {new_student["cgpa"]}', 0, 1)
    
    # Prediction result
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Placement Eligibility:', 0, 1)
    result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10,  result, 0, 1)
    count=0
    # Remedial subjects
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Remedial Subjects:', 0, 1)
    pdf.set_font('Arial', '', 12)
    if remedial_subjects:
        for subject in remedial_subjects:
            count+=1
        if count>=1:
                if new_student["computer_nw_score"]<20:
                    pdf.cell(0, 10, "Computer Networks", 0, 1)
                if new_student["software_engg_score"]<20:
                    pdf.cell(0, 10, "Software Engineering", 0, 1)
                if new_student["tfcs_score"]<20:
                    pdf.cell(0, 10, "Theoretical Foundations Of Computer Science[TFCS]", 0, 1)
                if new_student["pe2_score"]<20:
                    pdf.cell(0, 10,pe2_subject, 0, 1)
                if new_student["oe2_score"]<20:
                    pdf.cell(0, 10,oe2_subject, 0, 1)
                if new_student["oe3_score"]<20:
                    pdf.cell(0, 10,oe3_subject, 0, 1)
                if new_student["iks_score"]<20:
                    pdf.cell(0, 10,"Indian Knowledge System[IKS]", 0, 1)
             # Multi-line message
        pdf.set_font('Arial', '', 12)
        pdf.ln(10)  # Add some space before the multi-line message
        message = 'Try To Focus Above Subject(s).Try To Discuss The Subject(s) with Teachers.Try To Do Group Studies with Your Friends.'
        pdf.multi_cell(0, 10, message)
    else:
        pdf.cell(0, 10, 'None', 0, 1)
     # Centered multi-line message
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)  # Add some space before the multi-line message
    message = 'Happy Learning'
    message_width = pdf.get_string_width(message)
    pdf.set_x((pdf.w - message_width) / 2)
    pdf.multi_cell(0, 10, message)
    
    return pdf
# Function to generate PDF and return it as bytes
def generate_studentpdf(name,student_id,new_student, prediction, remedial_subjects1,pe2_subject,oe2_subject,oe3_subject):
    pdf = create_studentpdf(name,student_id,new_student, prediction, remedial_subjects1,pe2_subject,oe2_subject,oe3_subject)
    pdf_output = pdf.output(dest='S').encode('latin1')  # Generate PDF as bytes
    return pdf_output
# Function to create a PDF report
def create_teacherpdf(name,student_id,new_student, prediction,pe2_subject,oe2_subject,oe3_subject):
    pdf = FPDF()
    pdf.add_page()
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Teacher Assessed Report', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Name: "f'{name}', 0, 1)
    pdf.cell(0, 10, "USN: "f'{student_id}', 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Student Scores:', 0, 1)

    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Computer Networks: {new_student["computer_nw_score"]}', 0, 1)
    pdf.cell(0, 10, f'Software Engineering: {new_student["software_engg_score"]}', 0, 1)
    pdf.cell(0, 10, f'Theoretical Foundations Of Computer Science (TFCS): {new_student["tfcs_score"]}', 0, 1)
    pdf.cell(0, 10, f'{pe2_subject}: {new_student["pe2_score"]}', 0, 1)
    pdf.cell(0, 10, f'{oe2_subject}: {new_student["oe2_score"]}', 0, 1)
    pdf.cell(0, 10, f'{oe3_subject}: {new_student["oe3_score"]}', 0, 1)
    pdf.cell(0, 10, f'Indian Knowledge System (IKS): {new_student["iks_score"]}', 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Average Attendance of {name} is : {new_student["avg_attendance"]}', 0, 1)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Average Class Interest Predicted of {name} is : {prediction}', 0, 1)
     # Centered multi-line message
    pdf.set_font('Arial', 'B', 12)
    pdf.ln(10)  # Add some space before the multi-line message
    message = ':Happy Learning:'
    message_width = pdf.get_string_width(message)
    pdf.set_x((pdf.w - message_width) / 2)
    pdf.multi_cell(0, 10, message)
    
    return pdf
# Function to generate PDF and return it as bytes
def generate_teacherpdf(name,student_id,new_student, prediction, pe2_subject,oe2_subject,oe3_subject):
    pdf = create_teacherpdf(name,student_id,new_student, prediction,pe2_subject,oe2_subject,oe3_subject)
    pdf_output = pdf.output(dest='S').encode('latin1')  # Generate PDF as bytes
    return pdf_output

def main():
     # Navigation
    option = st.sidebar.selectbox("Select ", ["Student", "Teacher"])

    if option == "Student":
        st.header("Student")
        X, y = load_preprocess_student_data('studentmodel_dataset_csv.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, scaler = train_student_model(X_train, y_train)
        #accuracy, report = evaluate_model(model, scaler, X_test, y_test)

        st.write("Welcome, Student!")

        # Input fields
        name = st.text_input("Enter your name", "")
        student_id = st.text_input("Enter Your USN", "")

        st.subheader("Subject Marks")

        computer_nw_score= st.number_input('Enter Your Computer Networks Score', min_value=0, max_value=40,key='cn_score')

        software_engg_score= st.number_input('Enter Your Software Engineering Score', min_value=0, max_value=40,key='se_score')

        tfcs_score= st.number_input('Enter Your Theoretical Foundations Of Computer Science[TFCS] Score', min_value=0, max_value=40,key='tfcs_score')

        pe2_subject=st.selectbox('Select Your Professional Elective Subject', ['Choose your Subject','Big Data and Analytics', 'Data Mining', 'Advanced Algorithms','Cloud Computing'], key='pe2_subject')
        pe2_score= st.number_input(pe2_subject, min_value=0, max_value=40,key='pe2_score')

        oe2_subject=st.selectbox('Select Your Open Elective 2 Subject', ['Choose your Subject','Vehicular Systems', 'Environmental Technology','Public Health Technology','Occupational Health and Safety',
        'Sensor Technology','Image Processing','Electrical Safety For Engineers','Marketing Management','Engineering Economics',
        'Quality Control Engineering'], key='oe2_subject')
        oe2_score= st.number_input(oe2_subject, min_value=0, max_value=40,key='oe2_score')

        oe3_subject=st.selectbox('Select Your Open Elective 3 Subject', ['Choose your Subject','Electric Vehicles','Industrial Safety', 'Green Building Technology',
        'Disaster Management and Mitigation','Aircraft Electronics and System','Fuzzy Logic','Renewable Energy Sources',
        'Data Mining','Advanced Manufacturing Technology','Turbo Machines'], key='oe3_subject')
        oe3_score= st.number_input(oe3_subject, min_value=0, max_value=40,key='oe3_score')

        iks_score= st.number_input('Enter Your Indian Knowledge System[IKS] Score', min_value=0, max_value=40,key='iks_score')

        cgpa = st.number_input('Enter Your CGPA', min_value=0.0, max_value=10.0,step=0.01)

        if st.button("Submit"):
            if not name:
                st.warning("Name is required!")
            elif not student_id:
                st.warning("USN is required!")
            elif pe2_subject == 'Choose your Subject':
                st.warning("Professional Elective Subject Is Not Selected")
            elif oe2_subject == 'Choose your Subject':
                st.warning("Open Elective 2 Subject Is Not Selected")
            elif oe3_subject == 'Choose your Subject':
                st.warning("Open Elective 3 Subject Is Not Selected")
            else:
                # Example feature order from the original training data
                expected_feature_order = ['computer_nw_score', 'software_engg_score',
                                      'tfcs_score', 'pe2_score','oe2_score', 
                                      'oe3_score', 'iks_score', 'cgpa']
                # Prepare new student data for prediction
                new_student = {
                    'computer_nw_score': computer_nw_score,
                    'software_engg_score': software_engg_score,
                    'tfcs_score': tfcs_score,
                    'pe2_score': pe2_score,
                    'oe2_score': oe2_score,
                    'oe3_score': oe3_score,
                    'iks_score': iks_score,
                    'cgpa': cgpa}
                # Convert to DataFrame and ensure the columns are in the correct order
                new_student_df = pd.DataFrame([new_student])[expected_feature_order]

                # Now you can safely scale the new student data
                new_student_scaled = scaler.transform(new_student_df)

                # Predict placement eligibility
                prediction = model.predict(new_student_scaled)
                # Provide recommendations
                if prediction[0] == 1:
                    st.success('The student is eligible for the placement drive.')
                else:
                   st.success('The student is not eligible for the placement drive.')
                # Remedial classes and recommendations
                remedial_subjects = [subject for subject in new_student if 'score' in subject and new_student[subject] < 20]
                count=0
                if remedial_subjects:
                    st.write('The student needs remedial classes in the following subjects:')
                    for subject in remedial_subjects:
                        count+=1
                    if count>=1:
                        if new_student['computer_nw_score'] < 20:
                            st.write(f"Computer Networks")

                        if new_student['software_engg_score'] < 20:
                            st.write(f"Software Engineering")
                    
                        if new_student['tfcs_score'] < 20:
                            st.write(f"Theoretical Foundations Of Computer Science[TFCS]")
                    
                        if new_student['pe2_score'] < 20:
                            st.write(f"{pe2_subject}")

                        if new_student['oe2_score'] < 20:
                            st.write(f"{oe2_subject}")

                        if new_student['oe3_score'] < 20:
                            st.write(f"{oe3_subject}")

                        if new_student['iks_score'] < 20:
                            st.write(f"Indian Knowledge System")
                    st.write("Try To Focus Above Subject(s).\nTry To Discuss The Subject(s) with Teachers.\nTry To Do Group Studies with Your Friends.")
                else:
                    st.info('The student does not need remedial classes.')
                # Plotting the subjects' marks
                new_student1 = {
                'computer_nw_score': computer_nw_score,
                'software_engg_score': software_engg_score,
                'tfcs_score': tfcs_score,
                'pe2_score': pe2_score,
                'oe2_score': oe2_score,
                'oe3_score': oe3_score,
                'iks_score': iks_score,
                'cgpa': cgpa,
                'pe2_subject': pe2_subject,  # Example value for PE2 subject
                'oe2_subject': oe2_subject,  # Example value for OE2 subject
                'oe3_subject': oe3_subject  # Example value for OE3 subject
                }
                # Define subjects and their marks
                subjects = [
                'computer_nw_score', 
                'software_engg_score', 
                'tfcs_score', 
                'pe2_score', 
                'oe2_score', 
                'oe3_score', 
                'iks_score'
                ]
                marks = [new_student1[subject] for subject in subjects]

                # Replace the subject name for oe2_score with its value
                subject_labels = [
                    'CN' if subject == 'computer_nw_score' else
                    'SE' if subject == 'software_engg_score' else
                    'TFCS' if subject == 'tfcs_score' else
                    new_student1['pe2_subject'] if subject == 'pe2_score' else
                    new_student1['oe2_subject'] if subject == 'oe2_score' else
                    new_student1['oe3_subject'] if subject == 'oe3_score' else
                    'IKS' if subject == 'iks_score' else
                    subject
                    for subject in subjects
                ]

                # Create a bar chart
                fig, ax = plt.subplots()
                ax.bar(subject_labels, marks, color='blue')  # Plot the bar chart
                ax.set_xlabel('Subjects')  # Set x-axis label
                ax.set_ylabel('Marks')  # Set y-axis label
                ax.set_title('Marks Obtained in Each Subject')  # Set chart title
                plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

                # Display the chart in Streamlit
                st.pyplot(fig)
                 # Generate PDF
                pdf_bytes = generate_studentpdf(name,student_id,new_student1, prediction, remedial_subjects,pe2_subject,oe2_subject,oe3_subject)
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name=f"{student_id}_placement_report.pdf",
                    mime="application/pdf"
                )
                     
    elif option == "Teacher":
        st.header("Teacher")   
        X, y = load_preprocess_teacher_data('teachermodeldataset_csv.csv')
        # Splitting the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, scaler = train_teacher_model(X_train, y_train)
        
        st.write("Welcome, Teacher!")
        # Input fields
        name = st.text_input("Enter Student name", "")
        student_id = st.text_input("Enter Student USN", "")

        st.subheader("Subject Marks")

        computer_nw_score= st.number_input('Enter Computer Networks Score', min_value=0, max_value=40,key='cn_score')

        software_engg_score= st.number_input('Enter Software Engineering Score', min_value=0, max_value=40,key='se_score')
        
        tfcs_score= st.number_input('Enter  Theoretical Foundations Of Computer Science[TFCS] Score', min_value=0, max_value=40,key='tfcs_score')

        pe2_subject=st.selectbox('Select Professional Elective Subject', ['Choose Subject','Big Data and Analytics', 'Data Mining', 'Advanced Algorithms','Cloud Computing'], key='pe2_subject')
        pe2_score= st.number_input(pe2_subject, min_value=0, max_value=40,key='pe2_score')

        oe2_subject=st.selectbox('Select Open Elective 2 Subject', ['Choose Subject','Vehicular Systems', 'Environmental Technology','Public Health Technology','Occupational Health and Safety',
        'Sensor Technology','Image Processing','Electrical Safety For Engineers','Marketing Management','Engineering Economics',
        'Quality Control Engineering'], key='oe2_subject')
        oe2_score= st.number_input(oe2_subject, min_value=0, max_value=40,key='oe2_score')

        oe3_subject=st.selectbox('Select Open Elective 3 Subject', ['Choose Subject','Electric Vehicles','Industrial Safety', 'Green Building Technology',
        'Disaster Management and Mitigation','Aircraft Electronics and System','Fuzzy Logic','Renewable Energy Sources',
        'Data Mining','Advanced Manufacturing Technology','Turbo Machines'], key='oe3_subject')
        oe3_score= st.number_input(oe3_subject, min_value=0, max_value=40,key='oe3_score')

        iks_score= st.number_input('Enter Indian Knowledge System[IKS] Score', min_value=0, max_value=40,key='iks_score')

        #cgpa = st.number_input('Enter Students CGPA', min_value=0.0, max_value=10.0)

        avg_attendance = st.number_input('Enter Average Attendance', min_value=0.00, max_value=100.00)
        if st.button("Submit"):
            if not name:
                st.warning("Name is required!")
            elif not student_id:
                st.warning("USN is required!")
            elif pe2_subject == 'Choose Subject':
                st.warning("Professional Elective Subject Is Not Selected")
            elif oe2_subject == 'Choose Subject':
                st.warning("Open Elective 2 Subject Is Not Selected")
            elif oe3_subject == 'Choose Subject':
                st.warning("Open Elective 3 Subject Is Not Selected")
            else:
                def percentage_to_rating(percentage):
                    if percentage >= 91 and percentage<=100:
                        return 5
                    elif percentage >= 81 and percentage<=90:
                        return 4
                    elif percentage >= 71 and percentage<=80:
                        return 3
                    elif percentage >= 61 and percentage<=70:
                        return 2
                    elif percentage<=60:
                        return 1

                # Predict function
                def predict_class_interest(student_data):
                    # Calculate avg_score from the student's scores
                    avg_score = (student_data['computer_nw_score'] +
                                student_data['software_engg_score'] +
                                student_data['tfcs_score'] +
                                student_data['iks_score'] +
                                student_data['pe2_score'] +
                                student_data['oe2_score'] +
                                student_data['oe3_score']) / 280 * 100
                    
                    # Create input DataFrame for prediction
                    input_data = pd.DataFrame([{
                        'avg_attendance': student_data['avg_attendance'],
                        'avg_score': avg_score
                    }])
                    
                    # Ensure the input_data has the same columns as training data
                    input_data = input_data[['avg_attendance', 'avg_score']]
                    
                    # Standardize the input data
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    predicted_interest = model.predict(input_scaled)[0]
                    rating=percentage_to_rating(predicted_interest)
                    return rating
                # Example usage
                new_student = {
                    'computer_nw_score': computer_nw_score,
                    'software_engg_score': software_engg_score,
                    'tfcs_score': tfcs_score,
                    'iks_score': iks_score,
                    'pe2_score': pe2_score,
                    'oe2_score': oe2_score,
                    'oe3_score': oe3_score,
                    'avg_attendance':avg_attendance
                }

                predicted_interest = predict_class_interest(new_student)
                st.write(f"Predicted Average Class Interest: {predicted_interest}")
                 # Generate PDF
                pdf_bytes = generate_teacherpdf(name,student_id,new_student, predicted_interest,pe2_subject,oe2_subject,oe3_subject)
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name=f"teacher_assessed_report_{student_id}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
 
