# Movie Recommendation System

This project is a movie recommendation system built with Streamlit and powered by a machine learning model. It recommends movies based on a selected title's plot and genres using cosine similarity, and displays movie posters fetched from the TMDB API.
>YouTube tutorial [HERE](https://youtu.be/i-B_I2DGIAI)

>Dataset is [HERE](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

>Some files are not available on GitHub due to the large file size. you can get it [HERE](https://huggingface.co/sujoy0011/Movie-Recommendation-System/tree/main)

## Table of Contents

- [Overview](#overview)
- [Theory of Recommendation Systems](#theory-of-recommendation-systems)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The movie recommendation system helps users discover movies similar to their favorites. By selecting a movie from the list, users receive a list of top 10 recommended movies along with their posters. This project leverages machine learning techniques to analyze the features of movies and find similarities between them. It uses the TMDB API to fetch and display movie posters, enhancing the user experience by providing visual context for the recommendations. The system is built with Streamlit, providing an interactive and user-friendly interface for users to explore movie recommendations easily.

## Theory of Recommendation Systems

### What is a Recommendation System?

A recommendation system is a subclass of information filtering systems that seek to predict the rating or preference a user would give to an item. They are widely used in various applications like movie recommendations, product recommendations, and content recommendations.

### Types of Recommendation Systems

1. **Content-Based Filtering**: This method recommends items similar to those a user liked in the past. It relies on the attributes of the items and a profile of the user's preferences.

2. **Collaborative Filtering**: This method recommends items based on the preferences of similar users. It doesn't require the attributes of the items and instead focuses on user-item interactions.

3. **Hybrid Methods**: These methods combine content-based and collaborative filtering to provide more accurate recommendations.
   
    ![The-recommendation-system-types](https://github.com/user-attachments/assets/b08f4f84-9210-4dfb-9734-860b353a3da7)

### Cosine Similarity

In this project, cosine similarity compares TF-IDF vectors built from each movie's plot overview and genres. The similarity score indicates how alike the movie content is, rather than comparing the titles themselves.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Select a movie from the dropdown list and click "Find Similar Movies" to get similar movies along with their posters. Sequels and franchise movies are included by default; enable **Filter Sequels & Franchises** to exclude them.

## Dataset

The dataset used for this project contains information about movies, including their titles and IDs. It is processed and stored in `movie_data.pkl`. The dataset is used to calculate the cosine similarity between movies.

## Model

The model represents each movie using its plot overview and genres, then ranks movies by cosine similarity. Recommendations are filtered by shared genres and the selected minimum rating. Sequels and franchise movies can optionally be excluded.

## Results

The system provides the top 10 recommended movies for any selected movie title. It also fetches and displays the posters of these recommended movies using the TMDB API.

![Screenshot 2024-07-12 103743](https://github.com/user-attachments/assets/fbc357a1-a6e6-472a-892b-95fe96767743)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
