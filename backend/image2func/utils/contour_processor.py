import cv2
import numpy as np


def _calculate_contour_complexity(contour):
    """Calculate contour complexity based on curvature changes and point count."""
    contour = contour.reshape(-1, 2)
    n = len(contour)
    if n < 3:
        return 0

    # Count curvature changes
    complexity = 0
    for i in range(1, n - 1):
        v1 = contour[i] - contour[i - 1]
        v2 = contour[i + 1] - contour[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        # Higher weight for sharp turns (likely details like fingers, hair)
        if angle > 30:
            complexity += angle

    # Normalize by length
    length = cv2.arcLength(contour.astype(np.float32), closed=False)
    if length > 0:
        complexity = complexity / length * 100

    return complexity


def _calculate_center_distance(contour, img_width, img_height):
    """Calculate distance from contour centroid to image center (normalized)."""
    contour = contour.reshape(-1, 2)
    centroid = np.mean(contour, axis=0)
    center = np.array([img_width / 2, img_height / 2])
    dist = np.linalg.norm(centroid - center)
    max_dist = np.linalg.norm([img_width / 2, img_height / 2])
    return dist / max_dist if max_dist > 0 else 0


def _calculate_contour_score(contour, img_width, img_height, center_weight=0.3, complexity_weight=0.7):
    """Calculate priority score: higher = more likely to be person/important."""
    center_dist = _calculate_center_distance(contour, img_width, img_height)
    complexity = _calculate_contour_complexity(contour)
    length = cv2.arcLength(contour.astype(np.float32), closed=False)

    # Normalize length (longer contours in center are more likely to be person)
    max_possible_length = 2 * (img_width + img_height)
    length_score = min(length / max_possible_length * 10, 1.0)

    # Score: closer to center = higher, more complex = higher, longer = higher
    # Invert center distance (closer = higher score)
    center_score = 1.0 - center_dist

    # Combine scores
    score = (center_score * center_weight +
             complexity * complexity_weight +
             length_score * 0.3)

    return score


def extract_contours_from_array(img_bgr, min_contour_length=50, canny_threshold1=50, canny_threshold2=150,
                                prioritize_center=True, complexity_weight=0.7):
    """Extract contours directly from a BGR numpy array (no disk I/O).

    Args:
        prioritize_center: If True, sort contours by center proximity and complexity
        complexity_weight: Weight for complexity in scoring (0-1)
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image array")

    original = img_bgr.copy()
    height, width = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        if length >= min_contour_length:
            cnt = cnt.squeeze()
            if cnt.ndim == 2 and cnt.shape[1] == 2:
                filtered_contours.append(cnt.astype(np.float64))

    if prioritize_center:
        # Sort by combined score (center proximity + complexity)
        scored_contours = [
            (cnt, _calculate_contour_score(cnt, width, height,
                                           center_weight=0.3,
                                           complexity_weight=complexity_weight))
            for cnt in filtered_contours
        ]
        scored_contours.sort(key=lambda x: x[1], reverse=True)
        filtered_contours = [cnt for cnt, score in scored_contours]
    else:
        # Original sorting by length only
        filtered_contours.sort(key=lambda c: cv2.arcLength(c.astype(np.float32), closed=False), reverse=True)

    return filtered_contours, original, edges, width, height


def extract_contours(image_path, min_contour_length=50, canny_threshold1=50, canny_threshold2=150):
    """Extract contours from an image file path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return extract_contours_from_array(img, min_contour_length, canny_threshold1, canny_threshold2)


def sample_contour_uniform(contour, num_points=200):
    contour = contour.reshape(-1, 2)
    n = len(contour)
    if n <= num_points:
        return contour
    indices = np.linspace(0, n - 1, num_points, dtype=int)
    return contour[indices]


def get_contour_colors(contours, original_img, max_contours=50):
    """Extract average color for each contour from the original image."""
    colors = []
    h, w = original_img.shape[:2]
    for i, cnt in enumerate(contours):
        if i >= max_contours:
            break
        cnt = cnt.reshape(-1, 2).astype(np.int32)
        n = len(cnt)
        sample_indices = np.linspace(0, n - 1, min(20, n), dtype=int)
        sampled_colors = []
        for idx in sample_indices:
            x, y = int(cnt[idx][0]), int(cnt[idx][1])
            if 0 <= x < w and 0 <= y < h:
                bgr = original_img[y, x]
                rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
                sampled_colors.append(rgb)
        if sampled_colors:
            avg_r = int(np.mean([c[0] for c in sampled_colors]))
            avg_g = int(np.mean([c[1] for c in sampled_colors]))
            avg_b = int(np.mean([c[2] for c in sampled_colors]))
            colors.append(f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}")
        else:
            colors.append("#333333")
    return colors
