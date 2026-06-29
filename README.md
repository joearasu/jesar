For Activities 1, 3, and 4
Requirements for Running activity1_nuplan_enhanced_high_quality.py, activity2_nuplan_enhanced_high_quality.py, and activity3_nuplan_enhanced_high_quality.py 
1. Python Requirement

Use Python 3.10 or newer.
Python 3.11 is recommended for best compatibility and performance.

2. Required Python Packages

The script requires the following core packages:

py -m pip install numpy pandas

For PNG visualisation output, also install:

py -m pip install matplotlib seaborn

Please note that matplotlib and seaborn are only required for generating the optional PNG visualisation.

3. Dataset Requirements

The script can be run in two modes:

Demo Mode

Demo mode does not require any dataset. It generates sample data automatically.

py "C:\Users\joead\Downloads\CAV+IDS\activity1_nuplan_enhanced\activity1_nuplan_enhanced_high_quality.py" --demo
Real Dataset Mode

Real dataset mode supports the following file formats:

.csv
.db
.sqlite
.sqlite3
.gpkg for map context

Example command using the nuPlan Mini dataset and maps folder:

py "C:\Users\joead\Downloads\CAV+IDS\activity1_nuplan_enhanced\activity1_nuplan_enhanced_high_quality.py" `
  --mini-dataset "C:\Users\joead\Downloads\level5_cav_portfolio\nuplan-v1.1_mini\data\cache\mini" `
  --maps-dataset "C:\Users\joead\Downloads\level5_cav_portfolio\maps" `
  --max-datasets 5 `
  --skip-bad-datasets
4. Useful Command-Line Options

--mini-dataset
Specifies the scenario data file, folder, or multiple dataset files.

--maps-dataset
Specifies optional map or context data.

--mini-table
Specifies the SQLite table name if the .db file is not detected automatically.

--maps-table
Specifies the SQLite table name for map data, if required.

--output-dir
Specifies the folder where output results should be saved.

--simulation-size
Defines the number of simulated scenarios.

--max-datasets
Limits processing to the first selected number of dataset files.

--skip-bad-datasets
Allows the script to continue running even if one dataset fails to load.

--verbose-load
Displays detailed dataset loading messages.

--demo
Runs the script using generated demo data.

5. Output Files

When the script runs successfully, it creates the following files:

integrated_dataset.csv
simulation_results.csv
bayesian_risk_posterior.csv
policy_alternative_comparison.csv
fairness_audit_trail.csv
critical_evaluation_and_governance.md
activity1_critical_evaluation_report.html
governance_audit_log.jsonl
activity1_assessment_manifest.json

The script may also create the following optional PNG file:

activity1_nuplan_comprehensive_analysis.png

Important note: If matplotlib is not installed, the script will still run and generate the HTML report, CSV files, markdown report, and audit logs. However, it will skip the PNG visualisation.

6. Hardware Requirements

No GPU is required to run the script.

For large nuPlan .db files, it is recommended to start with the following option:

--max-datasets 5

This helps test the script on a smaller number of files first, reducing loading time and memory usage. Once the script runs successfully, the number of datasets can be increased.
