import requests

# url = 'http://localhost:9696/predict'
url = "https://favorite-song-prediction.fly.dev/predict"

customer = {
    "acousticness": 0.12,
    "danceability": 0.78,
    "duration_ms": 215000,
    "energy": 0.65,
    "instrumentalness": 0.00012,
    "liveness": 0.15,
    "loudness": -6.7,
    "speechiness": 0.045,
    "tempo": 123.5,
    "valence": 0.64,
    "song_title": "mask_off",
    "artist": "future",
    "key": 5,
    "mode": 1,
    "time_signature": 4
}


response = requests.post(url, json=customer)
predictions = response.json()

print(predictions)
if predictions['target']:
    print('customer like this song')
else:
    print('customer is not like this song')
