This project is build using PySpark with Python programming language and tested in macOS
Spark's MLlib libraries are used to perform the operations - https://spark.apache.org/mllib/

Requiremnts :
PySpark 3.11 or newer
Python 3.6 or newer
Anaconda

Installation:
https://docs.anaconda.com/anaconda/install/index.html
conda install -c conda-forge pyspark
sudo apt-get install python

Running the Pyspark Files:
Open the terminal
Activate the Anaconda environment : conda activate /Users/....
Then type - > pyspark

A way of executing Pyspark files is starting and creating a SparkSession in the terminal and feeding the commands as you go or by saving the file, navigating to the directory and running the file.

- The file in this project is a python script that can be run from the command line just navigating to the right directory and using the command:

python cancerdiagnosis.py

Once the command is run it the will automatically read the code, start a SparkSession and execute the code to produce the desired output.

REPORT

Implementation and Design of Machine Learning Algorithms using PySpark MLlib – https://spark.apache.org/mllib/

Logistic Regression and Random Forest for Cancer Diagnosis

Cancer diagnosis is a significant task in the medical field.
We approached this task using machine learning methods, the machine learning methods that I used are the Logistic Regression and Random Forest algorithms, to classify the instances as Bening or Malignant based on different features and to evaluate the performance of the selected model using the testing data.

Training Procedure:

The dataset was split into training (60%), validation (20%), and testing (20%) sets. The splitting process was deterministic by using a seed for reproducibility.
