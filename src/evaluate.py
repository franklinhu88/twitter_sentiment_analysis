from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_data
from sentiment_model import predict_sentiment

def evaluate_model():
    """Evaluates the sentiment analysis model on the dataset."""
    df = load_data()

    # Initialize counters for accurate predictions
    correct_predictions = 0
    total_predictions = len(df)

    # Get model predictions and print results as we process each tweet
    predicted_sentiments = []
    for idx, row in df.iterrows():
        tweet = row["Tweet content"]
        actual_sentiment = row["sentiment"]

        # Get prediction for this tweet
        predicted_sentiment = predict_sentiment(tweet)

        # Save the prediction
        predicted_sentiments.append(predicted_sentiment)

        # Check if the prediction is correct
        if predicted_sentiment.lower() == actual_sentiment.lower():
            correct_predictions += 1

        # Print progress every 100 rows
        if idx % 100 == 0:
            print(f"Processed {idx}/{total_predictions} tweets")

    # Add the predictions to the dataframe
    df["Predicted"] = predicted_sentiments

    # Check for any NaN values and drop them or fill them
    if df["sentiment"].isnull().any() or df["Predicted"].isnull().any():
        print("NaN values found. Dropping rows with NaN.")
        df = df.dropna(subset=["sentiment", "Predicted"])

    # Ensure all labels match expected format
    df["sentiment"] = df["sentiment"].str.strip().str.capitalize()
    df["Predicted"] = df["Predicted"].str.strip().str.capitalize()


    # Compute accuracy
    accuracy = accuracy_score(df["sentiment"], df["Predicted"])
    report = classification_report(df["sentiment"], df["Predicted"])

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    evaluate_model()
