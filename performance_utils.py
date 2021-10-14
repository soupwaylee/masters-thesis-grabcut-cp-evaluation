import numpy as np
import pandas as pd

import Cdf


def get_iou_for_mask_row(row, gt):
    image_id = row['image_id']
    i = np.logical_and(row['mask_arr'], gt[image_id])
    u = np.logical_or(row['mask_arr'], gt[image_id])
    return np.sum(i) / np.sum(u)


def get_prevalences_as_sample_avg_df(img_id_ordered, grabcut_masks_df, computed_masks_dict):
    image_id_to_masks_array = {}

    for id in img_id_ordered:
        gc_masks = np.stack(
            grabcut_masks_df.loc[grabcut_masks_df['image_id'] == id, 'mask_arr'].values,
            axis=0
        )
        seg_masks = computed_masks_dict[id][1:]
        masks = np.concatenate([seg_masks, gc_masks], axis=0)
        image_id_to_masks_array[id] = masks

    prevalences = [np.mean(image_id_to_masks_array[img_id]) for img_id in img_id_ordered]

    prevalences_df = pd.DataFrame.from_dict({
        'image_id': img_id_ordered,
        'prevalence': prevalences
    })
    return prevalences_df


def get_bias_column(annotator_performance_df):
    phi = annotator_performance_df.prevalence
    tpr = annotator_performance_df.sensitivity
    tnr = annotator_performance_df.specificity
    return tpr * phi + (1 - tnr) * (1 - phi)


def get_mcc_column(annotator_performance_df):
    beta = annotator_performance_df.bias
    phi = annotator_performance_df.prevalence
    tpr = annotator_performance_df.sensitivity
    tnr = annotator_performance_df.specificity
    return ((phi - phi ** 2) / (beta - beta ** 2)) ** 0.5 * (tpr + tnr - 1)


def get_ppv_column(annotator_performance_df):
    p = annotator_performance_df.sensitivity
    q = annotator_performance_df.specificity
    gamma = annotator_performance_df.prevalence
    return p * gamma / (p * gamma + (1 - q) * (1 - gamma))

def get_bias_for_mask_performance_row(row, prevalences_df):
    image_id = row['image_id']
    prevalence = prevalences_df.loc[prevalences_df['image_id'] == image_id, ['prevalence']].values[0]
    sensitivity = row['sensitivity']
    specificity = row['specificity']

    return sensitivity * prevalence + (1 - specificity) * (1 - prevalence)


def get_performance_cdf_df(annotator_performance_df, performance_value_name, category_name, categories):
    cdfs = []

    for cat in categories:
        vals = annotator_performance_df \
            .loc[annotator_performance_df[category_name] == cat][performance_value_name] \
            .values \
            .tolist()
        print(f'[*] {len(vals)} records for {category_name} == {cat}')
        cdfs.append(Cdf.MakeCdfFromList(vals))

    def cdf_to_concat_df(cdf, category):
        xs, ps = cdf.Render()
        category_list = [category for i in range(len(xs))]
        return pd.DataFrame.from_dict({
            'xs': xs,
            'ps': ps,
            category_name: category_list,
        })

    cdfs_df = pd.concat([cdf_to_concat_df(c, d) for d, c in zip(categories, cdfs)]).reset_index(drop=True)

    return cdfs, cdfs_df
