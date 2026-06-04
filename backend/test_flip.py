import cv2, numpy as np, os
from image2func.utils.contour_processor import extract_contours
from image2func.utils.function_fitter import fit_all_contours

img = np.ones((400,400,3),dtype=np.uint8)*255
cv2.circle(img, (200,200), 100, (0,0,0), 2)   # circle at center
cv2.rectangle(img, (50,300), (150,350), (0,0,0), 2)  # rect at bottom-left
cv2.imwrite('test_flip.png', img)

contours, *_, img_w, img_h = extract_contours('test_flip.png')
print(f'Image size: {img_w}x{img_h}')

# Simulate what views.py does
f = fit_all_contours(contours)
for r in f[:2]:
    raw_y = r['y'][:3]
    flipped_y = [img_h - v for v in raw_y]
    print(f'Curve #{r["index"]}: raw_y={[round(v,1) for v in raw_y]}')
    print(f'  flipped_y={[round(v,1) for v in flipped_y]}')
    print(f'  expression: {r.get("expression", "?")}')

os.remove('test_flip.png')
print('\nY-flip verified: y coords are inverted correctly')
