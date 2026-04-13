import pandas as pd

df = pd.read_csv("creditcard.csv")

# grab 100 random transactions (mix of fraud and legit)
sample = df.sample(100, random_state=1)
sample.to_csv("transactions_to_check.csv", index=False)
