# Streamify
Streamify is a streamlined alternative to Spotify, featuring a music recommendation system, playback, and streaming capabilities alongside real-time suggestions derived from user activity.

# Introduction

Spotify is a digital music streaming service providing access to millions of songs, podcasts, and videos from artists worldwide. It employs a sophisticated music recommendation system powered by machine learning algorithms. Your final assignment is to develop Streamify, a project that emulates Spotify's functionality while incorporating cutting-edge big data technologies.

# Extract, Transform, Load (ETL) Pipeline
In this phase, the focus was on creating an Extract, Transform, Load (ETL) pipeline using the Free Music Archive (FMA) dataset to prepare the data for subsequent processing and analysis.

## Objective
The primary objective of this phase was to establish a streamlined process for extracting data from the FMA dataset, transforming it into a suitable format, and loading it into a database for further utilization.

## Dataset
We utilized the fma_large.zip dataset, which consists of 106,574 tracks, each lasting 30 seconds. The metadata for these tracks, including details such as title, artist, genres, tags, and play counts, is available in fma_metadata.zip. This comprehensive dataset provided a rich source of information for our analysis and modeling tasks.

## Features
To extract meaningful information from the audio tracks, we employed various techniques such as Mel-Frequency Cepstral Coefficients (MFCC) using Python. These techniques allowed us to convert the audio files into numerical and vector formats, enabling further analysis and processing.

## Storage
After extracting the relevant audio features, we stored the transformed data in MongoDB, a NoSQL database known for its scalability and accessibility. MongoDB was chosen for its ability to handle large volumes of data efficiently, making it well-suited for storing the FMA dataset.

# Music Recommendation Model
In this phase, the objective was to train a music recommendation model using Apache Spark after storing the data in MongoDB. Various algorithms and methodologies were explored to achieve this goal.

## Objective
The primary objective of this phase was to develop a robust music recommendation model that could provide accurate and personalized recommendations to users based on their listening history and preferences.

## Technologies
Apache Spark was leveraged for its distributed computing capabilities, allowing for efficient processing of large-scale datasets. Additionally, MongoDB was utilized for storing the preprocessed music data in a scalable and accessible manner.

## Algorithms
Several algorithms were considered for building the recommendation model:
### Local Sensitive Hashing (LSH)
LSH is employed for approximate nearest neighbor search, which can be useful in recommendation systems for finding similar items or users efficiently. In the context of music recommendation, LSH can help identify tracks with similar audio features or user preferences.


# Deployment
In this phase, the objective was to deploy the trained music recommendation model onto a web application, specifically designed as a streaming service. The deployment process involves crafting an interactive and user-friendly web interface, integrating real-time music recommendations using Apache Spark, and ensuring seamless operation of the application.

## Web Application Development
The application provides an intuitive and visually appealing interface for users to browse, search, and play music seamlessly. 

## Real-Time Music Recommendations
To enhance user experience, Apache Spark will be leveraged to dynamically generate music recommendations in real-time. Historical playback data will be analyzed to provide personalized suggestions to users based on their preferences and listening habits. This recommendation engine will operate concurrently in the background of the web application, ensuring that users receive relevant and timely suggestions without any interruptions.

# Contribution

Aaleen Zainab  
Tooba Arshad  
Asma Riaz
