import os
import torch
import pandas as pd
import numpy as np
from PIL import Image

class OIDv6Dataset(object):
    def __init__(self, root, mode, mapping):
        """
        :param root: directory where data files are stored
        :param mode: one of 'train', 'test', 'validation'
        """
        self.mapping = mapping
        self.root = root
        self.mode = mode
        self.imgs = list(sorted(os.listdir(os.path.join(root, mode, 'images'))))
        self.annotation_df = pd.read_csv(os.path.join(root, 'cleaned_boxes', mode + '-annotations-bbox.csv'))


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.mode ,'images', self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_id = self.imgs[idx].split('.')[0]
        bboxes = self.annotation_df[self.annotation_df['ImageID'] == img_id].reset_index()
        boxes = bboxes[['XMin', 'YMin', 'XMax', 'YMax']].values
        area = (boxes[:,3] - boxes[:,1] * boxes[:,2] - boxes[:,0])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = np.zeros((bboxes.shape[0],))

        for i in range(bboxes.shape[0]):
            labels[i] = self.mapping[bboxes.loc[i, 'LabelName']]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        # target["boxes"] = torch.stack(list((map(torch.tensor, target['bozes'])))).type(torch.float32)
        return img, target


    def __len__(self):
        return len(self.imgs)



def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # use gpu if available, otherwise use cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    classes = ['accordion', 'cello', 'piano', 'saxophone', 'trumpet', 'violin']
    num_classes = len(classes)
    mapping = {'/m/0mkg': 0, '/m/01xqw': 1, '/m/05r5c': 2, '/m/06ncr': 3, '/m/07gql': 4, '/m/07y_7': 5}

    trainset = OIDv6Dataset('Dataset', 'train', mapping)

    data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=num_classes)
    model.to(device)



if __name__ == "__main__":
    main()
