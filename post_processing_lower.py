import cv2
import numpy as np
import random

def filter_predictions_by_confidence(predictions, threshold):
    return [pred for pred in predictions if pred['confidence'] >= threshold]

def polygons_are_same(poly1, poly2, threshold=10):
    center1 = np.mean([[p['x'], p['y']] for p in poly1], axis=0)
    center2 = np.mean([[p['x'], p['y']] for p in poly2], axis=0)
    return np.linalg.norm(center1 - center2) < threshold

def convert_to_hashable(obj):
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((k, convert_to_hashable(v)) for k, v in obj.items())
    return obj

def remove_duplicate_predictions(predictions, image_width):
    unique_predictions = []
    seen = set()
    for pred in predictions:
        duplicate_found = False
        for other_pred in predictions:
            if pred != other_pred and polygons_are_same(pred['points'], other_pred['points']):
                duplicate_found = True
                if pred['confidence'] > other_pred['confidence']:
                    pred_tuple = convert_to_hashable(pred)
                    if pred_tuple not in seen:
                        unique_predictions.append(pred)
                        seen.add(pred_tuple)
                else:
                    other_pred_tuple = convert_to_hashable(other_pred)
                    if other_pred_tuple not in seen:
                        unique_predictions.append(other_pred)
                        seen.add(other_pred_tuple)
                break
        if not duplicate_found:
            pred_tuple = convert_to_hashable(pred)
            if pred_tuple not in seen:
                unique_predictions.append(pred)
                seen.add(pred_tuple)
    return unique_predictions

def is_left_side(polygon, image_width):
    return np.mean([p['x'] for p in polygon]) < image_width / 2

def is_right_side(polygon, image_width):
    return np.mean([p['x'] for p in polygon]) >= image_width / 2

def correct_predictions(predictions, image_width, neighbor_threshold=50):
    corrected_classes = []
    for pred in predictions:
        pred_class = int(pred['class'])
        polygon = pred['points']

        if pred_class in range(41, 49) and is_right_side(polygon, image_width):
            neighbors = find_neighbors(pred, predictions, neighbor_threshold)
            corrected_class = infer_correct_class(pred_class, neighbors, 30)
            corrected_classes.append(corrected_class)
        elif pred_class in range(31, 39) and is_left_side(polygon, image_width):
            neighbors = find_neighbors(pred, predictions, neighbor_threshold)
            corrected_class = infer_correct_class(pred_class, neighbors, 40)
            corrected_classes.append(corrected_class)
        else:
            corrected_classes.append(pred_class)

        pred['class'] = corrected_classes[-1]
    return predictions

def find_neighbors(pred, predictions, neighbor_threshold):
    neighbors = []
    for other_pred in predictions:
        if other_pred != pred and is_neighbor(pred['points'], other_pred['points'], neighbor_threshold):
            neighbors.append(other_pred)
    return neighbors

def is_neighbor(polygon1, polygon2, threshold):
    dist = np.linalg.norm(np.mean([[p['x'], p['y']] for p in polygon1], axis=0) - np.mean([[p['x'], p['y']] for p in polygon2], axis=0))
    return dist < threshold

def infer_correct_class(pred_class, neighbors, group_offset):
    neighbor_classes = [int(n['class']) for n in neighbors if int(n['class']) in range(group_offset + 1, group_offset + 9)]

    if neighbor_classes:
        if group_offset == 30:
            corrected_class = min(neighbor_classes) - 10
        else:
            corrected_class = max(neighbor_classes) + 10
    else:
        corrected_class = pred_class - 10 if group_offset == 30 else pred_class + 10

    return corrected_class

def draw_predictions(image, predictions):
    overlay = image.copy()
    output = image.copy()

    for pred in predictions:
        points = np.array([(pt['x'], pt['y']) for pt in pred['points']], np.int32)
        points = points.reshape((-1, 1, 2))

        color = [random.randint(0, 255) for _ in range(3)]
        color = tuple(color)

        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=3)

        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        cv2.rectangle(overlay, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 10)

    for pred in predictions:
        points = np.array([(pt['x'], pt['y']) for pt in pred['points']], np.int32)
        points = points.reshape((-1, 1, 2))

        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        label = f"{pred['class']}"

        cv2.putText(overlay, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    alpha = 0.5
    output_image = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    output_path  = 'lower_predicted.jpg'
    cv2.imwrite(output_path, output_image)
    return output_image
