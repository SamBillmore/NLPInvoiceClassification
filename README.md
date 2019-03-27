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

As with all supervised machine learning classification models, the data used to train the model must be representative of the data required to be classified.

In addition, there are a number of required fields that the data must contain in order for the .pkl files to be created and for the application to run. 
