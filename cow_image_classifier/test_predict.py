from predict import predict_disease

image_path = "sample.jpg"

results = predict_disease(image_path)

print()

print("Prediction Result")

print()

print(f"Disease   : {results[0]['disease']}")

print(f"Confidence: {results[0]['confidence']} %")

print()

print("Top Predictions")

print()

for result in results:

    print(result)