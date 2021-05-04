import os
import pandas as pd

# Create new csv files containing only the annotations needed for training, validation and testing
# based on the classes in mapping

mapping = {'/m/0mkg': 0, '/m/01xqw': 1, '/m/05r5c': 2, '/m/06ncr': 3, '/m/07gql': 4, '/m/07y_7': 5}

for mode in ['train', 'test', 'validation']:
    if mode == 'train':
        annotation = 'oidv6-train'
    else:
        annotation = mode
    df = pd.read_csv('Dataset/boxes/' + annotation +'-annotations-bbox.csv')
    images = list(sorted(os.listdir('Dataset/' + mode + '/images')))
    img_ids = [imgs.split('.')[0] for imgs in images]

    new_df = df[(df['ImageID'].isin(img_ids)) & (df['LabelName'].isin(mapping.keys()))]
    new_df.to_csv('Dataset/cleaned_boxes/' + mode + '-annotations-bbox.csv', index=False)
