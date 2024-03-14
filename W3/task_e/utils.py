import pickle
import sys
import random

def save_custom_data(chosen_set, path_ids, path_labels):
    imgs_ids = []
    label_dict = {}
    for object_label in chosen_set:
        if object_label==str(89): print(chosen_set[object_label])
        for img_id in chosen_set[object_label]:
            imgs_ids.append(img_id)
            if img_id not in label_dict.keys():
                label_dict[img_id] = [object_label]
            else:
                inside_list = label_dict[img_id]
                inside_list.append(object_label)
                label_dict[img_id] = inside_list


    imgs_ids=list(set(imgs_ids))

    with open(path_ids, "wb") as f:
        pickle.dump(imgs_ids, f)

    with open(path_labels, "wb") as f:
        pickle.dump(label_dict, f)


# UTILS POS-NEG ====================================================================

def search_pos_neg(dict, curr_img_id, curr_labels):
    """
    This function is used to search for positive and negatives samples.
    Suppose we have:
        187464: ['49', '78', '79', '50', '82', '81', '44']
    Then we need to find a sample that does not contain any of the above labels (neg image)
    And a sample that contains the most of them (pos image)
    """
    pos_img_id = None
    neg_img_id = None
    max_common = 0
    min_common = float('inf')
    for img_id, labels in sorted(dict.items(), key=lambda x: random.random()):
        # Positive sample
        common_labels = list(set(curr_labels).intersection(labels))
        if len(common_labels) > max_common and img_id != curr_img_id:
            pos_img_id = img_id
            max_common = len(common_labels)
            if max_common == len(curr_labels): break
    
    pos_labels = dict[pos_img_id]
    curr_labels = curr_labels + pos_labels
    for img_id, labels in sorted(dict.items(), key=lambda x: random.random()):
        # Negative sample
        common_labels = list(set(curr_labels).intersection(labels))
        if len(common_labels) < min_common:
            neg_img_id = img_id
            min_common = len(common_labels)
            if min_common == 0: break
    return pos_img_id, neg_img_id


def create_image_loader(dict):
    """
    Will return a dataloader with:
        (anchor_id, pos_id, neg_id, anchor labels)
    """
    res = []
    for img_id, labels in tqdm.tqdm(dict.items()):
        # search for positive and negative samples
        pos_id, neg_id = search_pos_neg(dict, img_id, labels)
        res.append((img_id, pos_id, neg_id, labels))
    return res




    