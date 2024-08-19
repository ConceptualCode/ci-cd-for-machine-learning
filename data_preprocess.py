import re
import pandas as pd

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (#topic)
    text = re.sub(r'#\w+', '', text)
    
    # Remove all emojis (targeting most common emoji ranges)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF" 
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f" 
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special characters and digits (except word characters, spaces, and commas)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    
    return text

def preprocess_dataset(df, text_col='tweet', label_col='label'):
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].apply(clean_text)

    return df

if __name__ == "__main__":
    train_df = pd.read_csv('data/raw/shmuhammad_AfriSenti-twitter-sentiment/ibo_train.csv')
    test_df = pd.read_csv('data/raw/shmuhammad_AfriSenti-twitter-sentiment/ibo_test.csv')
    val_df = pd.read_csv('data/raw/shmuhammad_AfriSenti-twitter-sentiment/ibo_validation.csv')

    preprocessed_train_df = preprocess_dataset(train_df)
    preprocessed_test_df = preprocess_dataset(test_df)
    preprocessed_val_df = preprocess_dataset(val_df)

    preprocessed_train_df.to_csv('/home/tonyai/CICD-ML/preprocessed_data/train.csv', index=False)
    preprocessed_test_df.to_csv('/home/tonyai/CICD-ML/preprocessed_data/test.csv', index=False)
    preprocessed_val_df.to_csv('/home/tonyai/CICD-ML/preprocessed_data/val.csv', index=False)

    print("Preprocessing complete. Datasets saved.")