import csv
import fiftyone as fo
import pandas as pd
import numpy as np

# Small program to convert Open Images Dataset v6 to yolo format
# 
# source: https://medium.com/voxel51/loading-open-images-v6-and-custom-datasets-with-fiftyone-18b5334851c3

def create_detections(det_data, image_id, class_map):
    dets = []

    sample_dets = [i for i in det_data if i[0]==image_id]
    for sample_det in sample_dets:
        # sample_det reference: [ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside]
        label = class_map[sample_det[2]]
        xmin = float(sample_det[4])
        xmax = float(sample_det[5])
        ymin = float(sample_det[6])
        ymax = float(sample_det[7])

        # Convert to [top-left-x, top-left-y, width, height]
        bbox = [xmin, ymin, xmax-xmin, ymax-ymin]

        detection = fo.Detection(bounding_box=bbox, label=label)

        detection["IsOccluded"] = bool(int(sample_det[8]))
        detection["IsTruncated"] = bool(int(sample_det[9]))
        detection["IsGroupOf"] = bool(int(sample_det[10]))
        detection["IsDepiction"] = bool(int(sample_det[11]))
        detection["IsInside"] = bool(int(sample_det[12]))

        dets.append(detection)

    detections = fo.Detections(detections=dets)

    return detections


def main():

    # path to directory holding the cleaned up annotation files, i.e. only containing the annotations needed for this project
    # use cleandata.py to produce the cleaned up files
    path = 'Dataset/cleaned_boxes/'

    for mode in ['train', 'test', 'validation']:

        classes = ['accordion', 'cello', 'piano', 'saxophone', 'trumpet', 'violin']
        class_map = {'/m/0mkg': 'accordion', '/m/01xqw': 'cello', '/m/05r5c': 'piano', '/m/06ncr': 'saxophone', '/m/07gql': 'trumpet', '/m/07y_7': 'violin'}

        df = pd.read_csv(path + mode + '-annotations-bbox.csv')
        image_ids = df['ImageID'].unique()

        dataset = fo.Dataset("OIDv6")

        with open(path + mode + '-annotations-bbox.csv') as f:
            reader = csv.reader(f, delimiter=',')
            det_data = [row for row in reader]

        for image_id in image_ids.tolist():
            sample = fo.Sample(filepath='Dataset/' + mode + '/images/%s.jpg' % image_id)

            # Add Detections
            detections = create_detections(det_data, image_id, class_map)
            sample["detections"] = detections

            dataset.add_sample(sample)


        dataset.export(
            export_dir="Dataset/yolo_format/" + mode,
            dataset_type=fo.types.YOLODataset,
            label_field="detections",
        )


if __name__ == '__main__':
    main()
