import streamlit as st
import pandas as pd


# CSS
st.markdown(
    """
    <style>
    .streamlit-table {
        width: 100%;
        margin: 0 auto;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title('Main Menu')
section = st.sidebar.radio("Go to", [
    "Introduction",
    "K-means Clustering",
    "Linear Regression",
    "Descriptive Statistics & Visualization",
    "Apriori Algorithm"
])

# Introduction Section
if section == "Introduction":
    st.image('IE3_Cover.png', use_column_width=True)
    st.title('Stack Overflow Developer Survey Analysis Report')
    
    st.header('Introduction')
    st.write("""
    This report explores the Stack Overflow Developer Survey dataset, a comprehensive resource containing responses from developers worldwide. The dataset offers detailed insights into the global tech industry, focusing on job roles, skills, technologies used, and salary information. This dataset serves as a valuable tool for understanding the dynamics of the tech job market and analyzing trends across various developer demographics.
    Key attributes covered in the dataset include:

    - Demographic Information: Age, education level, country of residence, employment status, and job satisfaction.
    - Work Experience and Job Roles: Years of coding experience, job responsibilities, tools, and technologies used in professional settings.
    - Salary Information: Total compensation, job satisfaction ratings, and factors influencing earnings.
    - Technology Usage: Programming languages, platforms, databases, tools, and operating systems.
    - Artificial Intelligence (AI) Engagement: Developers’ usage of AI tools, challenges in AI adoption, and ethical considerations.

    By analyzing this data, we aim to explore how factors like job roles, geographic locations, years of experience, and technology usage impact developer salaries and satisfaction.
    """)
    
    st.markdown("---")
    
    # Checkbox to toggle visibility
if st.checkbox("Show Data Analysis Technique"):
    st.title("Data Analysis Techniques")
    
    st.markdown("""
    ## Overview
    This project applies several data analysis techniques to extract meaningful insights from the Stack Overflow Developer Survey dataset. Each technique is tailored to address specific research questions and explore trends in developer demographics, job roles, and compensation.
    """)

    # Expanders for detailed techniques
    with st.expander("a. Clustering (K-Means Clustering)"):
        st.markdown("""
        - **Purpose**: Group the data based on shared characteristics.
        - **Variables**:
          - Years of Experience: YearsCode, YearsCodePro
          - Job Role and Type: DevType, OrgSize, Industry
          - Salary Information: CompTotal
          - Technology Usage: Language, Database, Platform, ToolsTech, MiscTech
        - **Outcome**: Identify patterns and group developers with similar attributes for deeper analysis.
        """)

    with st.expander("b. Linear Regression"):
        st.markdown("""
        - **Purpose**: Predict developers’ salaries (CompTotal) based on independent variables.
        - **Key Variables**:
          - Years of Experience: YearsCodePro, YearsCode
          - Education Level: EdLevel
          - Industry: Industry
          - Job Satisfaction: JobSat, JobSatPoints
          - Technology Usage: Tools, platforms, and languages
        - **Outcome**: Quantify the influence of each factor on salary and identify key correlating aspects.
        """)

    with st.expander("c. Descriptive Statistics & Visualization"):
        st.markdown("""
        - **Purpose**: Summarize and visualize the data.
        - **Analysis**:
          - Salary distributions across regions, job roles, and experience levels.
          - Summary statistics for demographic variables (e.g., average age, coding experience, and job satisfaction).
        - **Visualizations**:
          - Histograms
          - Boxplots
          - Scatter plots
        """)

    with st.expander("d. Apriori Algorithm"):
        st.markdown("""
        - **Purpose**: Explore relationships between technology usage.
        - **Examples**:
          - Do developers using certain languages (e.g., Python, JavaScript) also use specific tools or frameworks?
          - Which technologies (e.g., cloud computing, databases) are frequently used together in job roles?
        - **Outcome**: Uncover patterns of technology adoption and tool combinations.
        """)

# K-means Clustering Section
if section == "K-means Clustering":
    st.title("K-means Clustering")
    st.write("Content for this section has been removed.")

# Linear Regression Section
if section == "Linear Regression":
    st.title("Linear Regression")
    st.write("Content for this section has been removed.")

# Descriptive Statistics & Visualization Section
if section == "Descriptive Statistics & Visualization":
    st.title("Descriptive Statistics & Visualization")
    st.write("Content for this section has been removed.")

# Apriori Algorithm Section
if section == "Apriori Algorithm":
    st.title("Apriori Algorithm")
    st.write("Content for this section has been removed.")
