import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import our data and drop unused columns. This should gives us something
# like this:
#   v1                                                 v2
#  ham  Go until jurong point, crazy.. Available only ...
#  ham                      Ok lar... Joking wif u oni...
# spam  Free entry in 2 a wkly comp to win FA Cup fina...
#  ham  U dun say so early hor... U c already then say...
#  ham  Nah I don't think he goes to usf, he lives aro...

data = pd.read_csv('datasets/ham-spam/spam.csv', encoding='latin-1')
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(index = str, columns = {'v1' : 'labels', 'v2' : 'text'})
# print(data.head())


# Split the dataset into two pieces and don't forget to reset
# the index. confirm the size and shape to be: ((4457, 2), (1115, 2))
train, test = train_test_split(data, test_size = 0.2, random_state = 42)
train.reset_index(drop=True), test.reset_index(drop=True)

print(train.shape)
print(test.shape)

# Finally, write this processed output to csv files:
train.to_csv('datasets/ham-spam/train.csv', index=False)
test.to_csv('datasets/ham-spam/test.csv', index=False)
