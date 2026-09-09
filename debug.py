import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/twcs.csv', nrows=100000)
amazon = df[df['author_id'] == 'AmazonHelp']
print(f'Amazon tweets: {len(amazon)}')
print(amazon[['tweet_id', 'in_response_to_tweet_id']].head(5))

customer_tweet_id = amazon['in_response_to_tweet_id'].dropna().iloc[0]
print(f'\nCustomer tweet id that amazon responded to: {customer_tweet_id}')
customer_tweet = df[df['tweet_id'] == customer_tweet_id]
if len(customer_tweet) > 0:
    row = customer_tweet.iloc[0]
    print(row)
    print(f"in_response_to_tweet_id type: {type(row['in_response_to_tweet_id'])}")
    print(f"str(val): {str(row['in_response_to_tweet_id'])}")
    
    tid = str(row['tweet_id'])
    print(f"tweet_id format: {tid}")
else:
    print('Customer tweet not found in first 100k rows')
