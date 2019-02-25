import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]

# Converts sex feature to numerical values: female = 1, male = 0
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Correlation between survival and sex
print(train_df['Survived'].corr(train_df['Sex']))

