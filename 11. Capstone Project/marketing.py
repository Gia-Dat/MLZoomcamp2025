import requests

url = 'http://localhost:8080/predict'

image_url = 'https://cataas.com/cat'

data = {'url': image_url}

print(f"Sending request to {url}...")

try:
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("\n--- Prediction Results ---")

        # Sort results by probability to show the top guess
        # (Assuming your service returns a dict of {class: prob})
        sorted_results = sorted(
            result.items(), key=lambda x: x[1], reverse=True)

        top_class, top_prob = sorted_results[0]
        print(f"Top Prediction: {top_class.upper()}")
        print(f"Confidence:     {top_prob*100:.2f}%")

        print("\nFull Breakdown:")
        for animal, prob in sorted_results[:3]:  # Show top 3
            print(f"- {animal:10}: {prob:.4f}")
    else:
        print(f"Error: Server returned status code {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Could not connect to the service: {e}")
