import pandas as pd

df = pd.read_csv('data/home-credit-default-risk/application_train.csv')

for i in range(2, 10):
    output_file = 'data/home-credit-default-risk/application_train_{}.csv'.format(i)
    print('Saving file: {}'.format(output_file))
    df['SK_ID_CURR'] = df['SK_ID_CURR'] + 1000000 * (i - 1)
    df.to_csv(output_file, index=False)
