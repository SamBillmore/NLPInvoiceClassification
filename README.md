# NLP Invoice Classification Flask application

An application to automatically classify invoices using Natural Language Processing and machine learning.

This Flask application was developed as a proof of concept to demonstrate the capability of classification algorithms to automate the classification of a large number of unclassified invoices.

All stages of the development are included in this repository, including:
- Code used in the initial data investigation and model development, including explanations of thought processes and approach (see model_dev directory)
- Code used to create the pipelines and pickle files (see pkl_creation directory)
- Code used in the Flask app, including the html and CSS (all code not in model_dev or pkl_creation directories)

Note: This repository does not contain the .pkl files required to run the Flask application due to file size limitations. However, these can be generated from any similar dataset using the relevant code in the pkl_creation directory. 

### Running the application

In order to run the application, two .pkl files will need to be created using the code in pkl_creation directory and some relevant training data.

**Data**

As with all supervised machine learning classification models, the data used to train the model must be representative of the data required to be classified. Also, the training data must have the classifications for each row of data, whereas the test data should not have the classifications.

The .pkl files should be created using the training data, which must contain the fields:
- Business Unit
- Supplier Name
- Supplier_Group 
- Invoice Desc 
- Invoice_Amt 
- Invoice Currency 
- USD_Amt 
- Project Owning Org 
- Datasource 
- Legacy 
- Year
- Leakage_Identifier 
- Leakage_Group 
- Intercompany_Flag 
- Americas_Flag
- Classification Group

**NB**: The data to be classified will not include the field Classification Group (as this is what the model will predict). 

Once the .pkl files are created, the application can be run. Save the two .pkl files in the same directory as the code for the Flask app - flask_app_v01.py (not in any sub-directory). Then navigate to that directory in the command prompt and run `python flask_app_v01.py`.

Copy and paste the generated url into Google Chrome (any other browser will work but the app may not be displayed as expected).

### Using the application

There are two demo options for running the model using the application: manual input of data into a web form; and upload a .csv file with the relevant columns. The tab to link to a database is a placeholder only.

Click on the required link in the header bar, enter ther required information or browse for the file to be uploaded, the click the button to classify the data.

### Code for initial model development

In order to provide a more useful overview of the entire process from initial data to final demo, I have included the code I used for investigating the data and testing different pre-processing steps and models, including commentary on my thought processes and approach.

Hopefully some might find this useful...

