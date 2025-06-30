import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

# Custom CSS Styling
custom_css = """
    <style>
        .stApp {
            background: linear-gradient(to right, #232526, #414345);
            font-family: 'Segoe UI', sans-serif;
            color: #ffffff;
        }

        h1.main-title {
            text-align: center;
            color: #00e676;
            font-size: 3rem;
            margin-top: 1rem;
        }

        p.sub-title {
            text-align: center;
            color: #bdbdbd;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        .stButton > button {
            background-color: #00e676;
            color: black;
            padding: 10px 24px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 16px;
        }

        .stButton > button:hover {
            background-color: #1de9b6;
            color: #000;
        }

        .book-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: transform 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }

        .book-card:hover {
            transform: scale(1.05);
        }

        .book-card p {
            color: #ffffff;
            font-weight: 500;
        }

        .selectbox-container {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 0 12px rgba(0, 255, 153, 0.1);
        }

        .custom-label {
            color: #ffffff !important;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            display: block;
        }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

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
            _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

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
                            <img src="{poster_url[i]}" width="100%" style="border-radius: 10px;" />
                            <p>{recommended_books[i]}</p>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            raise AppException(e, sys) from e

# UI Logic
if __name__ == "__main__":
    st.markdown("<h1 class='main-title'>📚 Book Recommendation Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Powered by collaborative filtering and wrapped in a stylish UI ✨</p>", unsafe_allow_html=True)

    obj = Recommendation()

    st.markdown("### 🔄 Train the Engine")
    if st.button('🚀 Train Recommender System'):
        obj.train_engine()

    st.markdown("---")
    book_names = pickle.load(open(os.path.join('artifacts/serialized_objects', 'book_names.pkl'), 'rb'))

    with st.container():
        st.markdown("<label class='custom-label'>📖 Select or type a book:</label>", unsafe_allow_html=True)
        selected_books = st.selectbox(
            label="", 
            options=book_names, 
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)


    if st.button('🎯 Show Recommendations'):
        obj.recommendations_engine(selected_books)
