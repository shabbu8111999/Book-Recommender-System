import os
import sys
import pickle
import pandas as pd
from books_recommender.logger.log import logging
from books_recommender.exception.exception_handler import AppException
from books_recommender.config.configuration import AppConfiguration


class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e
        

    def get_data_transformer(self):
        try:
            df = pd.read_csv(self.data_transformation_config.clean_data_file_path)

            # Let's create a Pivot table
            book_pivot = df.pivot_table(columns = 'User_id', index = 'Title', values = 'Book_rating')
            logging.info(f"Shape of Book Pivot Table: {book_pivot.shape}")
            book_pivot.fillna(0, inplace = True)

            # Saving the Pivot Table
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok = True)
            pickle.dump(book_pivot, open(os.path.join(self.data_transformation_config.transformed_data_dir, "transformed_data.pkl"), 'wb'))
            logging.info(f"Saved Pivot Table data to {self.data_transformation_config.transformed_data_dir}")

            # Kepping the Book names
            book_names = book_pivot.index

            # Saving book_names object for Web App
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok = True)
            pickle.dump(book_names, open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_names.pkl"), 'wb'))
            logging.info(f"Saved book_names Serialization object to {self.data_validation_config.serialized_objects_dir}")

            # Saving book_pivot Table for Web App
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok = True)
            pickle.dump(book_pivot, open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_pivot.pkl"), 'wb'))
            logging.info(f"Saved book_pivot Serialization object to {self.data_validation_config.serialized_objects_dir}")

        except Exception as e:
            raise AppException(e, sys) from e
        

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log Started. {'='*20} ")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log Completed. {'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e