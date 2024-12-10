import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules


@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_survey_results.csv')
    return data

data = load_data()
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
    "Data Preparation",
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

# Data Preparation Section
if section == "Data Preparation":
    st.title("Data Preparation")
    st.write("""
    In this section, we clean and prepare the data for analysis. This includes handling missing values, 
    dropping columns with excessive missing data, and imputing missing values where appropriate.
    """)

    # Show the first few rows of the original dataset
    st.subheader("Original Dataset Preview")
    st.write(data.head())

    # Display column information
    column_info = pd.DataFrame({
        'Column Name': data.columns,
        'Missing Values': data.isnull().sum(),
        'Data Type': data.dtypes
    }).reset_index(drop=True)

    st.subheader("Column Information")
    st.write(column_info)

    # Calculate the percentage of missing values
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    high_missing_cols = missing_percentage[missing_percentage > 50]
    
    # Display columns with more than 50% missing values
    st.subheader("Columns with More Than 50% Missing Values")
    if not high_missing_cols.empty:
        st.write(high_missing_cols)
    else:
        st.write("No columns have more than 50% missing values.")

    # Clean the data: drop columns with more than 50% missing values
    data_cleaned = data.drop(columns=high_missing_cols.index)
    st.write(f"Dataset shape after dropping columns: {data_cleaned.shape}")

    # Show remaining columns after cleaning
    st.subheader("Remaining Columns After Cleaning")
    remaining_columns = data_cleaned.columns
    st.write(remaining_columns)

    # Fill missing numerical values with the median
    numerical_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    data_cleaned[numerical_cols] = data_cleaned[numerical_cols].fillna(data_cleaned[numerical_cols].median())

    # Fill missing categorical values with the mode
    categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
    data_cleaned[categorical_cols] = data_cleaned[categorical_cols].fillna(data_cleaned[categorical_cols].mode().iloc[0])

    # Display missing values after imputation
    st.subheader("Missing Values After Imputation")
    missing_after_imputation = {
        'Numerical Columns': data_cleaned[numerical_cols].isnull().sum(),
        'Categorical Columns': data_cleaned[categorical_cols].isnull().sum()
    }
    st.write(missing_after_imputation)

    # Display the cleaned data preview
    st.subheader("Cleaned Dataset Preview")
    st.write(data_cleaned.head())

    # Save the cleaned dataset
    data_cleaned.to_csv('cleaned_survey_results.csv', index=False)

# K-means Clustering Section
if section == "K-means Clustering":
    st.title("K-means Clustering")
    st.write("Content for this section has been removed.")

# Linear Regression Section
if section == "Linear Regression":
    st.title("Linear Regression")

    # Display the explanation
    st.write("""
    This linear regression model predicts developers' salaries (CompTotal) using key factors like years of experience (YearsCode, YearsCodePro), education level (EdLevel), and job role (DevType). 
    The data is cleaned by converting categorical experience values to numeric and encoding categorical features using one-hot encoding. 
    The model is trained on 80% of the data and tested on 20% to evaluate performance using Mean Squared Error (MSE) and R-squared (R²). 
    The most influential features are identified, highlighting which factors have the strongest impact on salary, providing insights into how experience, education, and role affect developer compensation.
    """)

    try:
        # Data Cleaning (already done but included here for clarity)
        data = data.copy()

        # Handle 'YearsCode' and 'YearsCodePro' conversion
        data['YearsCode'] = data['YearsCode'].replace({'Less than 1 year': 0.5, 'More than 50 years': 50}).apply(pd.to_numeric, errors='coerce')
        data['YearsCodePro'] = data['YearsCodePro'].replace({'Less than 1 year': 0.5, 'More than 50 years': 50}).apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing target or independent variables
        data = data.dropna(subset=['CompTotal', 'YearsCode', 'YearsCodePro', 'EdLevel', 'DevType'])

        # One-hot encode categorical columns
        data = pd.get_dummies(data, columns=['EdLevel', 'DevType'], drop_first=True)

        # Select relevant columns for the regression
        X = data[['YearsCode', 'YearsCodePro'] + [col for col in data.columns if 'EdLevel_' in col or 'DevType_' in col]]
        y = data['CompTotal']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Get the coefficients and their corresponding features
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        # Display the results
        st.subheader("Model Evaluation")
        st.write(f'Mean Squared Error (MSE): {mse:.2f}')
        st.write(f'R-squared (R²): {r2:.2f}')

        st.subheader("Top 10 Most Influential Features")
        st.write(coefficients.head(10))

    except Exception as e:
        st.error(f"Error loading or processing the data: {e}")

# Descriptive Statistics & Visualization Section
if section == "Descriptive Statistics & Visualization":
    st.title("Descriptive Statistics & Visualization")
    st.write("Content for this section has been removed.")

# Apriori Algorithm Section
if section == "Apriori Algorithm":
    st.title('Apriori Algorithm for')
