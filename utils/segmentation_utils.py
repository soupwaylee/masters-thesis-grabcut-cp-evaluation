import cv2 as cv
import numpy as np


def clean_phase_image(
        img,
        min_intensity=20,
        min_size=10,
        min_max_opt_height=80,
        min_mean_opt_height=80):
    img[img <= min_intensity] = 0
    result = cv.connectedComponentsWithStats(img, connectivity=8)
    component_total_count, components_mask, component_stats, component_centroids = result

    sizes = component_stats[1:, -1]
    # print(sizes)
    component_total_count = component_total_count - 1

    img_cleaned = np.zeros(img.shape)

    for i in range(component_total_count):
        if (sizes[i] >= min_size
            and (np.mean(img[components_mask == i + 1]) >= min_mean_opt_height
                 or np.max(img[components_mask == i + 1]) >= min_max_opt_height)):
            img_cleaned[components_mask == i + 1] = img[components_mask == i + 1]

    return img_cleaned.astype(np.uint8)


def adaptive_threshold_segmentation(
        img,
        filter_size=5,
        block_size=21,
        c=0):
    """
    Calculates the binary image using adaptive threshold.
    Author: ga78luv
    filterSize -– kernel size for the used median filter
    blockSize –- Size of a pixel neighborhood that is used to calculate a threshold value for the pixel (musst be an odd integer)
    C -– Constant subtracted from the mean or weighted mean
    """
    blurred = cv.medianBlur(img, filter_size)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, c)
    thresh = fill_holes(thresh)  # Fill holes in the latter mask!
    return thresh


def fill_holes(img):
    """
    Author: ga78luv
    """
    # Method to fill holes potentially resulting from applying adaptive thresholding.
    # Copy the thresholded image.
    floodfill = img.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    floodfill_inverted = cv.bitwise_not(floodfill)

    # Combine the two images to get the foreground.
    img_out = img | floodfill_inverted
    return img_out


def box_segmentation(
        images,
        threshold_value=50,
        min_size=30,
        margin=2,
        box_size=(50, 50),
        min=0.0,
        max=4.0,
):
    """Returns a list of masks where the bounding boxes describe the one-valued foreground,
    otherwise zero-valued background """
    masks = []

    for img in images:
        # multiplier = 255.0 / (max - min)
        # img_int = ((np.clip(img, min, max) - min) * multiplier).astype(np.uint8)
        contours, threshold = find_contours(img, threshold_value, min_size)
        boxes = find_bounding_boxes(contours, margin, box_size)
        new_mask = np.zeros(img.shape, np.uint8)
        for box in boxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            new_mask[y1:y2, x1:x2] = 1
        masks.append(new_mask)

    return masks


def global_threshold_segmentation(
        images,
        threshold_value=50,
        min_size=30,
        margin=2,
        box_size=(50, 50),
        min=0.0,
        max=4.0,
        filter_oof_cells=False,
        return_oof_cells=False,
):
    cell_imgs = []
    masks = []
    relative_contours = []
    gradients = []
    oof_cells = []

    for img in images:
        # multiplier = 255.0 / (max - min)
        # img_int = ((np.clip(img, min, max) - min) * multiplier).astype(np.uint8)
        contours, threshold = find_contours(img, threshold_value, min_size)
        boxes = find_bounding_boxes(contours, margin, box_size)

        for contour, box in zip(contours, boxes):
            if cmp_tuple(box[0], (0, 0), "ge") and cmp_tuple(box[1], img.shape[::-1], "le"):
                # Cut out the part of the image that contains the cell
                image_patch = cut_image_patch(img, box)

                # Calculate the image gradient
                gx, gy = np.gradient(image_patch)
                gradient = np.sqrt(np.square(gx) + np.square(gy))

                # Filter the oof cells
                if filter_oof_cells:
                    contour_grad, _ = find_contours(
                        gradient.astype(np.float32), 0.2, min_size
                    )
                    if len(contour_grad) != 0:
                        area_factor = cv.contourArea(
                            contour_grad[0]
                        ) / cv.contourArea(contour)
                        if 0.8 > area_factor > 0.01:
                            oof_cells.append(image_patch)
                            continue

                # Cut out the part of the image that contains the cell
                cell_imgs.append(image_patch)

                # Cut out the part of the binary mask that contains the cell
                masks.append(cut_image_patch(threshold, box))

                # Set the origin of the contour in the upper left corner of the bounding box
                relative_contours.append(contour - box[0])

                # append gradient patch
                gradients.append(gradient)

    cell_imgs = np.stack(cell_imgs)
    masks = np.stack(masks)
    gradients = np.stack(gradients)

    if filter_oof_cells:
        print(
            f"[ INFO ] Omitted {len(oof_cells)} of {len(cell_imgs) + len(oof_cells)} cells ({len(oof_cells) / (len(oof_cells) + len(cell_imgs)) * 100:.2f}%) due to bad focus!"
        )

    if return_oof_cells:
        oof_cells = np.stack(oof_cells)
        return cell_imgs, masks, relative_contours, gradients, oof_cells

    return cell_imgs, masks, relative_contours, gradients


def find_contours(image, threshold_value, min_size):
    _, threshold = cv.threshold(image, threshold_value, 1, cv.THRESH_BINARY)
    try:
        _, contours, _ = cv.findContours(
            threshold.astype(np.uint8),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )
    except ValueError:
        # Ensures compatibility with other versions of opencv
        contours, _ = cv.findContours(
            threshold.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
    # Store all contours bigger than the threshold in the contours attribute list
    contours = [c for c in contours if cv.contourArea(c) > min_size]
    # Sort the list ascended according the the contour size
    contours.sort(key=cv.contourArea)
    return contours, threshold


def find_bounding_boxes(contours, margin, box_size):
    boxes = []
    if box_size is None:
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            boxes.append([(x - margin, y - margin), (x + w + margin, y + h + margin)])
    else:
        for contour in contours:
            moments = cv.moments(contour)
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            x1 = int(cx - np.ceil(box_size[0] / 2))
            y1 = int(cy - np.ceil(box_size[1] / 2))
            x2 = x1 + box_size[0]
            y2 = y1 + box_size[1]
            boxes.append([(x1, y1), (x2, y2)])
    return boxes


def cmp_tuple(x, y, relation):
    if relation == "l":
        return all([(a < b) for a, b in zip(x, y)])
    elif relation == "le":
        return all([(a <= b) for a, b in zip(x, y)])
    elif relation == "g":
        return all([(a > b) for a, b in zip(x, y)])
    elif relation == "ge":
        return all([(a >= b) for a, b in zip(x, y)])


def cut_image_patch(image, box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    return image[y1:y2, x1:x2]
