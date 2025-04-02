from sentiment_model import predict_sentiment

def main():
    """Runs the interactive sentiment analysis agent."""
    print("Twitter Sentiment Analysis Agent")
    print("Type 'exit' to quit.\n")

    while True:
        tweet = input("Enter a tweet: ")
        if tweet.lower() == "exit":
            break

        sentiment = predict_sentiment(tweet)
        print(f"Predicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
