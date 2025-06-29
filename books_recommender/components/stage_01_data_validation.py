import os
import sys
import ast
import pandas as pd
import pickle
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException



class DataValidation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    
    def preprocess_data(self):
        try:
            rating_df = pd.read_csv(self.data_validation_config.ratings_csv_file, sep=";", on_bad_lines='skip', encoding='latin-1')
            book_df = pd.read_csv(self.data_validation_config.books_csv_file, sep=";", on_bad_lines='skip', encoding='latin-1')
            
            logging.info(f" Shape of ratings data file: {rating_df.shape}")
            logging.info(f" Shape of books data file: {book_df.shape}")

            # Will select only the Large URL column because it's resolution is perfect
            book_df = book_df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
            
            # Re-naming my column to use it easily
            book_df.rename(columns = {"Book-Title":"Title",
                          "Book-Author":"Author",
                          "Year-Of-Publication":"Year",
                          "Image-URL-L":"Image_url"}, inplace = True)

            
            # Lets remane some wierd columns name in ratings
            rating_df.rename(columns = {"User-ID":"User_id",
                            "Book-Rating":"Book_rating"}, inplace = True)

            # Lets store users who had at least rated more than 200 books
            x = rating_df['User_id'].value_counts() > 200
            y = x[x].index
            rating_df = rating_df[rating_df['User_id'].isin(y)]

            # Now join ratings with books
            rating_with_book = rating_df.merge(book_df, on = 'ISBN')
            num_of_rating = rating_with_book.groupby('Title')['Book_rating'].count().reset_index()
            num_of_rating.rename(columns = {"Book_rating":"no_of_rating"}, inplace = True)
            final_rating = rating_with_book.merge(num_of_rating, on = 'Title')

            # Let's take those books which got only 50 or above 50 ratings
            final_rating = final_rating[final_rating['no_of_rating'] >= 50]

            # dropping the duplicates
            final_rating.drop_duplicates(inplace = True)
            logging.info(f" Shape of the final clean dataset: {final_rating.shape}")
                        
            # Saving the cleaned data for transformation
            os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
            final_rating.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'clean_data.csv'), index = False)
            logging.info(f"Saved cleaned data to {self.data_validation_config.clean_data_dir}")


            #saving final_rating objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(final_rating,open(os.path.join(self.data_validation_config.serialized_objects_dir, "final_rating.pkl"),'wb'))
            logging.info(f"Saved final_rating serialization object to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e

    
    def initiate_data_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.preprocess_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e