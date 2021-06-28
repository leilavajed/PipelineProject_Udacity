import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    
    """
    load the messages and categories datasets and merge them them together.
    Returns:
    a df : merged datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages,categories,on='id')

def is_notone(inputs):
    if inputs == '2':
        return '1'
    else:
        return inputs
def clean_data(df):
    """
    cleans dataset and extracts classification categories.
    parameters:
        df: dataset containing messages and categories
    Returns:
        df: cleaned dataset
    """    
        
    categories = df['categories'].str.split(";",expand=True,)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x:is_notone(x.split('-')[1]))
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis=1)
    df =  pd.concat([df,categories],sort=False ,axis=1)
    return df.drop_duplicates()


def save_data(df, database_filename):
    """ save the clean dataset into an sqlite database in the provided path """
    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()