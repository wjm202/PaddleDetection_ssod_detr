import json
roidbs = json.load(
    open('dataset/coco/annotations/instances_train2017.json', 'r'))
roidbsfull_img_ids = [db['image_id'] for db in roidbs["annotations"]]
supervised_idx = json.load(open('dataset/coco/COCO_supervision.txt', 'r'))[str(
    10.0)][str(1)]
labeled_idx = []
for x in supervised_idx:
    if x in set(roidbsfull_img_ids):
        labeled_idx.append(x)
        print(x)
# labeled_idx = [x for x in supervised_idx if x in set(roidbsfull_img_ids)]
print("1")

unlabeled_idx = [
    id for id in (full_img_ids) if id not in list(set(labeled_idx))
]
print("1")
