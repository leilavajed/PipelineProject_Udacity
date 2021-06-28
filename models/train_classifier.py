import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier 
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """ load data from sqlitd db """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name=engine.table_names()[0],con='sqlite:///'+database_filepath)
    X = df['message'].values
    Y = df[(df.columns[4:])].values
    return X,Y,df.columns[4:]


def tokenize(text):
    """ exchange url with url placeholder, lemmatizing and tokenizing the input(text)"""
    
             
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    the pipeline take in the 'message' column as input and output classification result
    use dSearch to find better parameters to tune the model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 4)] 
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    parameters = { 'clf__estimator__max_features': max_features,
                   'clf__estimator__max_depth': max_depth,
                   'clf__estimator__min_samples_split': min_samples_split,
                   'clf__estimator__min_samples_leaf': min_samples_leaf,
                   'clf__estimator__bootstrap': bootstrap}


    cv = GridSearchCV(pipeline, param_grid=parameters ,cv = 3)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """Report the overall accuracy and a classification report for each output category of the dataset. 
       parameters : model,X_test,Y_test, category_names
       output: print classification_report
        classification reports: 'precision','recall','f1-score', 'support'
       """
    y_pred = model.predict(X_test)  
    for i in range(len(Y_test)):
        print(classification_report(y_pred=y_pred[:,i],y_true=Y_test[:,i]))
    


def save_model(model, model_filepath):
    """
     Export our model as a pickle file
     parameters: model, model_filepath
    
     """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        """ split data into train and test datasets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()