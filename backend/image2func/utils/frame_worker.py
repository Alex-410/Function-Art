"""Worker module for parallel frame processing. Must be importable for multiprocessing."""
import cv2
from .contour_processor import extract_contours_from_array, get_contour_colors
from .function_fitter import fit_all_contours


def process_single_frame(args):
    """Process a single video frame. Called by ProcessPoolExecutor."""
    frame_rgb, img_w, img_h, min_contour_length, smoothing, max_contours = args
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    contours, original_img, edges_img, w, h = extract_contours_from_array(
        frame_bgr, min_contour_length=min_contour_length
    )
    fitted_results = fit_all_contours(contours, smoothing=smoothing, max_contours=max_contours)
    colors = get_contour_colors(contours, original_img, max_contours=max_contours)

    curve_data = []
    for i, r in enumerate(fitted_results):
        xs = r['x']
        ys = [img_h - v for v in r['y']]
        curve_data.append({
            'index': r['index'],
            'num_points': r.get('num_points', 0),
            'expression': r.get('expression', 'Complex Curve'),
            'color': colors[i] if i < len(colors) else '#333333',
            'x': [round(v, 2) for v in xs],
            'y': [round(v, 2) for v in ys],
        })

    bounds = {
        'x_min': 0,
        'x_max': img_w,
        'y_min': 0,
        'y_max': img_h,
    }

    return {
        'curves': curve_data,
        'bounds': bounds,
        'total_contours': len(contours),
        'fitted_contours': len(fitted_results),
    }
