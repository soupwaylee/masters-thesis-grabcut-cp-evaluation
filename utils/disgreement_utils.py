import numpy as np
import pandas as pd
import SimpleITK as sitk


def get_masks(grabcut_masks_df, computed_masks_dict, image_id):
    masks_per_image_df = grabcut_masks_df.loc[grabcut_masks_df['image_id'] == image_id, ['mask_arr', 'mask_uuid', 'academic_field']]

    annotator_masks = masks_per_image_df['mask_arr'].values
    annotator_masks_uuids = masks_per_image_df['mask_uuid'].values
    annotator_masks = np.stack(annotator_masks, axis=0)

    computed_masks = computed_masks_dict[image_id][1:]

    return computed_masks, annotator_masks, annotator_masks_uuids


def get_staple_outputs(computed_masks, annotator_masks, annotator_masks_uuids, img_id=None):
    computers = computed_masks.shape[0]
    annotators = annotator_masks.shape[0]

    all_masks = [computed_masks[i].astype(np.uint8) for i in range(computers)] \
                + [annotator_masks[i].astype(np.uint8) for i in range(annotators)]

    sitk_images = [sitk.GetImageFromArray(mask) for mask in all_masks]

    staple_filter = sitk.STAPLEImageFilter()
    seg_estimate = staple_filter.Execute(sitk_images)

    print(f"[*] {img_id:<15} {staple_filter.GetElapsedIterations()} STAPLE iterations")

    # seg_estimate = sitk.GetArrayViewFromImage(result)
    sensitivities = staple_filter.GetSensitivity()
    specificities = staple_filter.GetSpecificity()

    computed_mask_scores = (sensitivities[:computers], specificities[:computers])
    annotator_mask_scores = (sensitivities[computers:], specificities[computers:], annotator_masks_uuids.tolist())

    return seg_estimate, computed_mask_scores, annotator_mask_scores


def get_staple_eval_data(img_id_ordered, grabcut_masks_df, computed_masks_dict):
    masks = {}
    staple_eval_data = {}

    for img_id in img_id_ordered:
        computed_masks, annotator_masks, annotator_masks_uuids = get_masks(grabcut_masks_df, computed_masks_dict, img_id)
        masks[img_id] = {
            'computed_masks': computed_masks,
            'annotator_masks': annotator_masks,
        }

        staple_outputs = get_staple_outputs(computed_masks, annotator_masks, annotator_masks_uuids, img_id)
        staple_eval_data[img_id] = {
            'staple_estimate': staple_outputs[0],
            'computed_mask_scores': {
                'sensitivities': staple_outputs[1][0],
                'specificities': staple_outputs[1][1],
            },
            'annotator_mask_scores': {
                'sensitivities': staple_outputs[2][0],
                'specificities': staple_outputs[2][1],
                'mask_uuid': staple_outputs[2][2],
            },
        }

    return masks, staple_eval_data


def build_annotator_performance_df(img_id_ordered, staple_eval_data):
    annotator_performance_data = []

    for img_id in img_id_ordered:
        sensitivities = staple_eval_data[img_id]['annotator_mask_scores']['sensitivities']
        specificities = staple_eval_data[img_id]['annotator_mask_scores']['specificities']
        mask_uuids = staple_eval_data[img_id]['annotator_mask_scores']['mask_uuid']

        for sensitivity, specificity, uuid in zip(sensitivities, specificities, mask_uuids):
            annotator_performance_data.append([img_id, sensitivity, specificity, uuid])

    annotator_performance_df = pd.DataFrame(annotator_performance_data, columns=['image_id', 'sensitivity', 'specificity', 'mask_uuid'])
    imgs = pd.api.types.CategoricalDtype(ordered=True, categories=img_id_ordered)
    annotator_performance_df['image_id'] = annotator_performance_df['image_id'].astype(imgs)

    return annotator_performance_df