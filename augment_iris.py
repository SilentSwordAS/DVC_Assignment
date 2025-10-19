import pandas as pd
import sys



df_1 = pd.read_csv(f'{sys.argv[1]}.csv')
df_2 = pd.read_csv(f'{sys.argv[2]}.csv')
df = pd.concat([df_1, df_2])

df.to_csv(f'augmented_train_{sys.argv[3]}.csv', index=False)
