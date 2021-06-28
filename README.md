# Disaster Response Pipeline Project from Udacity
## Introduction
When a disaster occurs, we usually see many posts on social media and news stations that either inform the people, warn, or request for help.

In this project,I will create a ML model that analyzes and classify messages from real disaster data, [Figure Eight](https://appen.com/), into different categories depending on their meaning. The model will be used by a web application to display the result. An emergency worker can input a new message through the web app and get classification results in several categories

This project can be used to collect messages that people post on different social media platforms when a disaster occurs such floods, storms, or fire and analyze the current situation. It can be helpful by speeding up the time of response of rescue teams and news stations. Also it can help by warning nearby people of the occurance of a disaster and help them evacuate quickly. Finally, it will allow rescue teams to prioritize their actions based on the situation.
#### File Structure

- app<br>
| - template<br>
| |- master.html  # main page of web app<br>
| |- go.html  # classification result page of web app<br>
|- run.py  # Flask file that runs app<br>

- data<br>
|- disaster_categories.csv  # data to process<br> 
|- disaster_messages.csv  # data to process<br>
|- process_data.py<br>
|- CleanDatabase.db   # database to save clean data to<br>

- models<br>
|- train_classifier.py<br>
|- cv_AdaBoostr.pkl  # saved model <br>

- README.md<br>

## Project structure:


1. ETL Pipeline: In a Python script, process_data.py,  a data cleaning pipeline was written that:

   - Loads the messages and categories datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database
2. ML Pipeline: In a Python script, train_classifier.py,  a machine learning pipeline was written that:

   - Loads data from the SQLite database
   - Splits the dataset into training and test set
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file
3. Flask Web App : Udacity provided much of the flask web app for us. For this part, I did folowing tasks:

   - Modify file paths for database and model as needed
   - Add data visualizations using Plotly in the web app. One example is provided for you
        1- template(1-1- master.html # main page of web app . 1-2- go.html # classification result page of web app)

        2- run.py # Flask file that runs app


## Requirements:
To get the flask app working you need:
- python3
- packages in requirements.txt file

install the packages by running:
- `pip3 install wheel`
- `pip3 install -r requirements.txt`

## Instructions:
1. Run the following commands to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database, **go to the data directory** and run:  
            `python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`  

    - To run ML pipeline that trains classifier and saves the model, **go to the models directory** and run:  
             `python3 train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Run the following command in the **app directory** to run the web app.  
         `python3 run.py`

3. Go to http://0.0.0.0:3001/