import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


# Custom CSS styling for the whole app
def local_css():
    st.markdown("""
        <style>
        /* Main app background */
        .stApp {
            background-color: #1e1e1e;
            font-family: 'Segoe UI', sans-serif;
            color: #ffffff;
        }

        h1, h2, h3, h4 {
            color: #ffffff;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #10a37f;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
        }

        .stButton>button:hover {
            background-color: #13b58c;
            color: #ffffff;
        }

        /* Selectbox container */
        .selectbox-container {
            background-color: #2c2c2c;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.6);
            margin-bottom: 20px;
        }

        /* Recommendation card */
        .book-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            text-align: center;
            transition: 0.3s ease;
        }

        .book-card:hover {
            transform: scale(1.03);
        }

        /* Text */
        .book-card p {
            color: #ffffff;
            font-weight: bold;
        }

        /* Header text */
        .main-title {
            text-align: center;
            color: #10a37f;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .sub-title {
            text-align: center;
            color: #a0a0a0;
            font-size: 18px;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)



class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]:
                ids = int(np.where(final_rating['Title'] == name)[0][0])
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['Image_url']
                poster_url.append(url)

            return poster_url

        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            book_id = int(np.where(book_pivot.index == book_name)[0][0])
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

            poster_url = self.fetch_poster(suggestion)

            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list, poster_url

        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            with st.spinner("Training model, please wait..."):
                obj = TrainingPipeline()
                obj.start_training_pipeline()
            st.success("✅ Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)
            cols = st.columns(5)

            for i in range(1, 6):
                with cols[i - 1]:
                    st.markdown(f"""
                        <div class='book-card'>
                            <img src="{poster_url[i]}" width="100%" style="border-radius: 8px;" />
                            <p style="margin-top:10px;"><strong>{recommended_books[i]}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            raise AppException(e, sys) from e


# -------------------- Streamlit UI Starts --------------------

if __name__ == "__main__":
    local_css()

    st.markdown("<h1 class='main-title'>📚 Book Recommendation Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>A sleek, AI-powered system built with Collaborative Filtering</p>", unsafe_allow_html=True)

    obj = Recommendation()

    # Training Button
    st.markdown("### 🔄 Train the Engine")
    if st.button('🚀 Train Recommender System'):
        obj.train_engine()

    st.markdown("---")

    # Book Select Box Styled Container
    book_names = pickle.load(open(os.path.join('artifacts/serialized_objects', 'book_names.pkl'), 'rb'))
    with st.container():
        selected_books = st.selectbox("📖 Select or type a book:", book_names)
        st.markdown("</div>", unsafe_allow_html=True)

    # Recommendations
    if st.button('🎯 Show Recommendations'):
        obj.recommendations_engine(selected_books)
