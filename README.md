# Analysis of Volume Bid Contracts
Custom Application to Perform Analysis of Volume Bid Contract Data vs Analysis of Historical Contract Data


## Table of contents

- [Research Question](#research_question)
- [Project Components](#project_components)
- [Project Data](#project_data)
- [Processed Data](#processed_data)
- [Project Notebooks](#project_notebooks)
- [Project Documents](#documents_notebooks)
- [Python Application](#python_application)
- [Environment](#environment)
- [Running Application](#running_application)
- [Git Hub Repo](#git_Hub_repo)

----------------------------------------------------------------------


## Research Question
This project evaluates how implementing volume bid contracts affects service-level agreements for data center equipment maintenance. Specifically, it examines whether using volume bid contracts increases the average repair times for equipment in data centers compared to traditional contracting methods.

----------------------------------------------------------------------


## Project Components
This project includes the following files and folders:
- README.md – This file contains an overview of the project application.
- Requirements.txt – This file contains a list of all the external packages and libraries that are necessary for the project.
- Environment.yml – This file contains the dependencies necessary to create an environment with conda.
- Data – This directory contains the current and historical CSV data files.
- Processed Data – This directory contains the data after processing has been completed.
- Notebooks – This directory has the Jupyter notebook with the output from the Python application.
- Documents – This directory contains PDF and HTML versions of the Jupyter notebook.
- Python – This directory contains the ten Python files that comprise the application.

----------------------------------------------------------------------


## Project Data
The data directory contains the two CSV files necessary for the project analysis.
- Work_Completed_Set1.csv
- Work_Completed_Set2.csv

### Source of Data
The project uses simulated data consisting of two separate datasets in CSV format. The simulated data was created using custom Python scripts within a series of Jupyter Notebooks. The first dataset contains historical information simulating the previous contract cycle, including a wide range of vendors and totaling 28,385 records. The second dataset simulates data from the current contract cycle with the volume bid vendor group, comprising 36,906 records. Each dataset has 13 columns that capture key attributes such as coverage area, work category, vendor, priority, date submitted, date completed, and due date.

### Work Completed Set 1

| Column            | Data Type | Null Values | Non-Null Values | Unique Values |
|-------------------|:---------:|:-----------:|-----------------|---------------|
| work_order_id     | string    | 0           | 28385           | 28385         |
| work_order_status | string    | 0           | 28385           | 1             |
| region            | string    | 0           | 28385           | 1             |
| coverage_area     | string    | 0           | 28385           | 4             |
| availability_zone | string    | 0           | 28385           | 18            |
| data_center       | string    | 0           | 28385           | 52            |
| work_category     | string    | 0           | 28385           | 11            |
| sub_category      | string    | 0           | 28385           | 106           |
| vendor            | string    | 0           | 28385           | 90            |
| priority          | string    | 0           | 28385           | 10            |
| date_submitted    | string    | 0           | 28385           | 26609         |
| date_completed    | string    | 0           | 28385           | 26866         |
| due_date          | string    | 0           | 28385           | 19389         |

### Work Completed Set 2

| Column            | Data Type | Null Values | Non-Null Values | Unique Values |
|-------------------|:---------:|:-----------:|-----------------|---------------|
| work_order_id     | string    | 0           | 36906           | 36906         |
| work_order_status | string    | 0           | 36906           | 1             |
| region            | string    | 0           | 36906           | 1             |
| coverage_area     | string    | 0           | 36906           | 4             |
| availability_zone | string    | 0           | 36906           | 18            |
| data_center       | string    | 0           | 36906           | 53            |
| work_category     | string    | 0           | 36906           | 10            |
| sub_category      | string    | 0           | 36906           | 85            |
| vendor            | string    | 0           | 36906           | 40            |
| priority          | string    | 0           | 36906           | 10            |
| date_submitted    | string    | 0           | 36906           | 34868         |
| date_completed    | string    | 0           | 36906           | 35404         |
| due_date          | string    | 0           | 36906           | 35910         |

----------------------------------------------------------------------


## Processed Data
The processed data directory holds the CSV file generated by the data processing. The new dataset contains 65,291 entries across 23 columns, including 13 original features and 10 engineered features.
- Work_Completed_Data.csv

### Work Completed Data

| Column            | Column Type        | Data Type | Null Values | Non-Null Values | Unique Values |
|-------------------|--------------------|:---------:|:-----------:|-----------------|---------------|
| work_order_id     | original feature   | string    | 0           | 65291           | 65291         |
| work_order_status | original feature   | string    | 0           | 65291           | 1             |
| region            | original feature   | string    | 0           | 65291           | 1             |
| coverage_area     | original feature   | string    | 0           | 65291           | 4             |
| availability_zone | original feature   | string    | 0           | 65291           | 18            |
| data_center       | original feature   | string    | 0           | 65291           | 54            |
| work_category     | original feature   | string    | 0           | 65291           | 11            |
| sub_category      | original feature   | string    | 0           | 65291           | 123           |
| vendor            | original feature   | string    | 0           | 65291           | 104           |
| priority          | original feature   | string    | 0           | 65291           | 10            |
| date_submitted    | original feature   | string    | 0           | 65291           | 61477         |
| date_completed    | original feature   | string    | 0           | 65291           | 62270         |
| due_date          | original feature   | string    | 0           | 65291           | 55299         |
| dataset_number    | engineered feature | integer   | 0           | 65291           | 2             |
| dataset_name      | engineered feature | string    | 0           | 65291           | 2             |
| dateSubmitted     | engineered feature | datetime  | 0           | 65291           | 61477         |
| dateCompleted     | engineered feature | datetime  | 0           | 65291           | 62270         |
| dueDate           | engineered feature | datetime  | 0           | 65291           | 55299         |
| hours_to_complete | engineered feature | float     | 0           | 65291           | 28294         |
| days_past_due     | engineered feature | float     | 0           | 65291           | 13268         |
| completed_on_time | engineered feature | boolean   | 0           | 65291           | 2             |
| sla_group         | engineered feature | integer   | 0           | 65291           | 2             |
| sla_label         | engineered feature | string    | 0           | 65291           | 2             |

----------------------------------------------------------------------


## Project Notebooks
The notebooks directory contains the Jupyter notebook used to run the Python application. The notebook is organized into 10 main sections, each dedicated to a specific part of the analysis process. There are two versions of the notebook: the full version and the demo version. The full version includes the code and all comments, while the demo version has the code with all comments removed.
- Analysis of Volume Bid Contracts.ipynb
- Analysis of Volume Bid Contracts – Demo.ipynb

----------------------------------------------------------------------


## Documents
The documents folder holds PDF and HTML versions of the Jupyter notebook. These files show a copy of the analysis results and the application's output in a clear, easy-to-view format.
- Analysis of Volume Bid Contracts.pdf
- Analysis of Volume Bid Contracts.html

----------------------------------------------------------------------


## Python Application
The Python directory contains the twelve Python files that comprise the project application.
1. **greetings.py** - This file contains the greetings function that returns the customary ‘Hello World!’ greeting.
2. **process_data.py** – This file defines the ProcessData class for uploading individual data files, integrating them into a comprehensive dataset, and conducting feature engineering to facilitate thorough data analysis.
3. **visual_analysis.py** – This file defines the VisualAnalysis class for conducting a thorough visual examination of the data using histograms, box plots, bar plots, violin plots, and KDE plots. Additionally, it includes performing descriptive statistical analyses and presenting the findings in a tabular format.
4. **outlier_analysis.py** – This file defines the OutlierAnalysis class for performing a thorough analysis of the data to identify outliers and potential anomalies, with the findings presented using box plots, histograms, and comprehensive statistical tables.
5. **distribution_analysis.py** – This file defines the DistributionAnalysis class for conducting a distribution analysis on the data to assess its normality using both the Shapiro-Wilk and Anderson-Darling tests. The evaluation also includes an analysis of skewness and kurtosis.
6. **transformation_analysis.py** - This file defines the TransformationAnalysis class for performing a comprehensive analysis of the data by applying five established transformation techniques to find the most suitable method for the data. The results are shown through histograms, Q-Q plots, and statistical tables.
7. **t_test_analysis.py** – This file defines the TTestAnalysis class for implementing Independent Samples T-Tests on the data to evaluate statistical significance by comparing current data with historical data. The results are displayed using box plots, violin plots, histograms, and tabular summaries.
8. **u_test_analysis.py** – This file defines the UTestAnalysis class for conducting a non-parametric analysis of the data using the Mann-Whitney U Test to evaluate statistical differences between current and historical data, with results presented as box plots, violin plots, histograms, ECDF plots, and tabular summaries.
9. **parametric_nonparametric_analysis.py** – This file defines the ParametricNonparametricAnalysis class for evaluating the effectiveness of parametric and non-parametric tests applied to the data and determining the most appropriate analytical approach, with results systematically presented in tabular form.
10. **chi_square_analysis.py** – This file defines the ChiSquareAnalysis class for performing a Chi-Squared Analysis on the data to identify any statistically significant relationships between attributes. The analysis results are displayed using a heatmap visualization and summarized in detailed statistical tables.
11. **data_analysis_main.py** – The file defines the DataAnalysisMain class for implementing the data processing and the eight analysis classes.
12. **test_analysis.py** – This file defines the unit tests for the Python application.

----------------------------------------------------------------------


## Environment
The environment can be set up using either pip or conda.
- Option 1: Use the supplied file `environment.yml` to create a new environment with conda
- Option 2: Use the supplied file `requirements.txt` to create a new environment with pip

### Create Environment
If conda is installed and ready, a new environment can be created using the ``environment.yml``
file provided in the root of the repository by executing the following code:

```bash
> conda env create -f environment.yml
> conda activate sla_analysis
```

----------------------------------------------------------------------


## Running Application
To run the application:
1. Open up a terminal and navigate to the directory holding the project folder, and enter the following commands:

```bash
> cd Analysis_of_Volume_Bid_Contracts
> cd Notebooks
> jupyter notebook
```

2. Once the Jupyter home page opens up, click on `Analysis of Volume Bid Contracts.ipynb` to start the notebook.
3. After the notebook starts, click on the double arrow on the toolbar at the top to restart the Kernel and run all cells.

----------------------------------------------------------------------


## Git Hub Repo
https://github.com/K-Irwin-WGU/Analysis-of-Volume-Bid-Contracts.git
