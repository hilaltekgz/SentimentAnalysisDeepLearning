import pandas as pd
from termcolor import colored
from sklearn.model_selection import train_test_split

# Define variables
COLUMNS = ["username","acctdesc","location","following","followers","totaltweets","usercreatedts","tweetcreatedts","retweetcount","text","hashtags","Unnamed: 0","favourite_count","retweet_count","created_at","sentiment","sentiment-class","tweet","Clean_tweet"]

# Read dataset
dataset = pd.read_csv(r'C:\Users\Msı\Desktop\YL\Derin_Ogrenme\Twitter-Sentiment-Analysis\clean_train.csv', names = COLUMNS, encoding = 'latin-1')
print(colored("Columns: {}".format(', '.join(COLUMNS)), "yellow"))

# Remove extra columns
print(colored("Useful columns: Sentiment and Tweet", "yellow"))
print(colored("Removing other columns", "red"))
dataset.drop(["username","acctdesc","location","following","followers","totaltweets","usercreatedts","tweetcreatedts","retweetcount","text","hashtags","Unnamed: 0","favourite_count","retweet_count","created_at"], axis = 1, inplace = True)
print(colored("Columns removed", "red"))

# Train test split
print(colored("Splitting train and test dataset into 80:20", "yellow"))
X_train, X_test, y_train, y_test = train_test_split(dataset['Clean_tweet'], dataset['sentiment-class'], test_size = 0.20, random_state = 100)
train_dataset = pd.DataFrame({
	'tweet': X_train,
	'sentiment-class': y_train
	})
print(colored("Train data distribution:", "yellow"))
print(train_dataset['sentiment-class'].value_counts())
test_dataset = pd.DataFrame({
	'tweet': X_test,
	'sentiment-class': y_test
	})
print(colored("Test data distribution:", "yellow"))
print(test_dataset['sentiment-class'].value_counts())
print(colored("Split complete", "yellow"))

# Save train data
print(colored("Saving train data", "yellow"))

train_dataset.to_csv('train.csv', index = False)
print(colored("Train data saved to train.csv", "green"))

# Save test data
print(colored("Saving test data", "yellow"))
test_dataset.to_csv('test.csv', index = False)
print(colored("Test data saved to test.csv", "green"))
