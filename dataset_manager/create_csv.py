import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_NAME = 'mountaindogs'                                           # provide name of folder containing images in subfolder 'data'

dataset_path = os.path.join('../data/datasets', DATASET_NAME)           # path to dataset folder
data_path = os.path.join(dataset_path, 'data')                          # path to images

filenames = os.listdir(data_path)
labels = [file.split('-')[0] for file in filenames]                     # extract label from image name

df = pd.DataFrame({'filename': filenames, 'label': labels})             # create dataframe containing filenames and corresponding label

train, res = train_test_split(df, test_size=0.3, stratify=df['label'])          # train test split is actually not needed for evaluation but is needed to run author code
val, test = train_test_split(res, test_size=0.67, stratify=res['label'])

train['subset'], val['subset'], test['subset'] = 'train', 'val', 'test'         # add subset column needed to run author code

df = pd.concat([train, val, test], ignore_index=True)
df.to_csv(os.path.join(dataset_path, f'{DATASET_NAME}.csv'))
