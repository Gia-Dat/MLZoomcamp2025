# Midterm Project

## Problems

- I want to know what is the song that be liked or not by the listener on Spotify so we can send them the recommended song based on the information of that song
- I took the data from Kaggle and then do some EDA and train my model: 
dataset link "https://www.kaggle.com/datasets/geomack/spotifyclassification"

## Solving

### Model Training
- I trained the model with notebook.ipynb
- What I did is that I did some EDA and then train the dataset with multiple models, then I tuning all of these model to see what is the best fit for this data
-> I found that Random Forest is the best fit

### Exporting to script
- Then turned this to the script
- And wrote train.py, predict.py, marketing.py this one is what we send to the customer

### Create environment
- Then for dependencies and environment: I run these code
  - uv init
  - rm main.py
  - uv add scikit-learn fastapi uvicorn
  - uv add --dev requests
  - uv sync

### Containerization
- After that I create a Dockerfile (see in my Dockerfile), build it: docker build -t predict-churn .

### Cloud deployment
- For cloud deployment I used Fly.io, I used Windows so that these are my code:
  - flyctl auth login
  - flyctl launch
  - flyctl deploy

Then I have a link and put it in marketing.py
with "/predict" after that link, and the final link to my deployment:
https://favorite-song-prediction.fly.dev/predict

<img width="2559" height="1360" alt="image" src="https://github.com/user-attachments/assets/4d6f8203-c352-4d74-9b53-9c94c6bc00ae" />

