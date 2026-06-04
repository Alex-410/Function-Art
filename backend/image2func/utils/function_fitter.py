import numpy as np
from scipy.interpolate import splprep, splev
from scipy import ndimage


def fit_contour_spline(contour, smoothing=5):
    contour = contour.reshape(-1, 2)
    n = len(contour)
    if n < 4:
        return _fit_poly(contour)

    mask = np.ones(n, dtype=bool)
    mask = _simplify_by_angle(contour, mask, angle_threshold=15.0)
    mask = _simplify_by_distance(contour, mask, min_dist=2.0)

    simplified = contour[mask]
    ns = len(simplified)

    if ns < 4:
        simplified = contour
        ns = n

    tck, u = splprep([simplified[:, 0], simplified[:, 1]], s=smoothing, per=False)

    u_fine = np.linspace(0, 1, 300)
    x_fine, y_fine = splev(u_fine, tck)

    return {
        "tck": tck,
        "u": u_fine.tolist(),
        "x": x_fine.tolist(),
        "y": y_fine.tolist(),
        "control_points": simplified.tolist(),
    }


def _fit_poly(contour):
    contour = contour.reshape(-1, 2)
    t = np.linspace(0, 1, len(contour))
    px = np.polyfit(t, contour[:, 0], min(3, len(contour) - 1))
    py = np.polyfit(t, contour[:, 1], min(3, len(contour) - 1))
    u_fine = np.linspace(0, 1, 300)
    x_fine = np.polyval(px, u_fine)
    y_fine = np.polyval(py, u_fine)
    return {
        "type": "poly",
        "u": u_fine.tolist(),
        "x": x_fine.tolist(),
        "y": y_fine.tolist(),
    }


def _simplify_by_angle(contour, mask, angle_threshold=15.0):
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return mask
    for i in range(1, n - 1):
        if not mask[i]:
            continue
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle < angle_threshold:
            mask[i] = False
    return mask


def _simplify_by_distance(contour, mask, min_dist=2.0):
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return mask
    kept_indices = np.where(mask)[0]
    if len(kept_indices) < 2:
        return mask
    new_mask = np.zeros(n, dtype=bool)
    new_mask[kept_indices[0]] = True
    last_kept = kept_indices[0]
    for idx in kept_indices[1:]:
        dist = np.linalg.norm(pts[idx] - pts[last_kept])
        if dist >= min_dist:
            new_mask[idx] = True
            last_kept = idx
    return new_mask


def _format_poly(var, coeffs):
    terms = []
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        p = deg - i
        if abs(c) < 1e-6:
            continue
        if p == 0:
            terms.append(f"{c:.4f}")
        elif p == 1:
            terms.append(f"{c:.4f}{var}")
        else:
            terms.append(f"{c:.4f}{var}^{p}")
    if not terms:
        return "0"
    expr = " + ".join(terms)
    expr = expr.replace("+ -", "- ")
    return expr


def _fit_fourier_series(t, data, n_terms=8):
    """Fit data with Fourier series: a0 + sum(ak*cos(2*pi*k*t) + bk*sin(2*pi*k*t))"""
    n = len(t)
    if n < 4:
        return None
    a0 = np.mean(data)
    coeffs = []
    for k in range(1, n_terms + 1):
        ak = 2 * np.mean(data * np.cos(2 * np.pi * k * t))
        bk = 2 * np.mean(data * np.sin(2 * np.pi * k * t))
        if abs(ak) > 1e-6 or abs(bk) > 1e-6:
            coeffs.append((k, ak, bk))
    return a0, coeffs


def _format_fourier(a0, coeffs, var):
    """Format Fourier series expression."""
    parts = []
    if abs(a0) > 1e-6:
        parts.append(f"{a0:.4f}")
    for k, ak, bk in coeffs[:6]:
        if abs(ak) > 1e-6:
            parts.append((ak, f"cos(2π·{k}{var})"))
        if abs(bk) > 1e-6:
            parts.append((bk, f"sin(2π·{k}{var})"))
    if not parts:
        return "0"
    expr = ""
    for i, item in enumerate(parts):
        if isinstance(item, str):
            expr = item
        else:
            val, fn = item
            sign = " + " if val >= 0 else " - "
            expr += f"{sign}{abs(val):.4f}·{fn}"
    expr = expr.replace("+ -", "- ")
    expr = expr.replace("- +", "- ")
    return expr.strip()


def generate_expression(contour, fitted_result):
    try:
        pts = contour.reshape(-1, 2)
        if len(pts) < 4:
            return "Simple Curve (insufficient points)"
        x, y = pts[:, 0], pts[:, 1]

        x_unique_ratio = len(np.unique(np.round(x))) / len(x)
        if x_unique_ratio > 0.6:
            for deg in range(1, 6):
                if deg >= len(x):
                    break
                coeffs = np.polyfit(x, y, deg)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
                if r2 > 0.85:
                    expr = _format_poly("x", coeffs)
                    return f"y = {expr}  (R²={r2:.3f})"

        t = np.linspace(0, 1, len(x))
        for deg in range(1, 5):
            if deg >= len(x) // 2:
                break
            px = np.polyfit(t, x, deg)
            py = np.polyfit(t, y, deg)
            x_pred = np.polyval(px, t)
            y_pred = np.polyval(py, t)
            errors = np.sqrt((x - x_pred) ** 2 + (y - y_pred) ** 2)
            if np.mean(errors) < 5:
                x_expr = _format_poly("t", px)
                y_expr = _format_poly("t", py)
                return f"x(t) = {x_expr}  ;  y(t) = {y_expr}"

        fx = _fit_fourier_series(t, x, n_terms=8)
        fy = _fit_fourier_series(t, y, n_terms=8)
        if fx is not None and fy is not None:
            x_expr = _format_fourier(fx[0], fx[1], "t")
            y_expr = _format_fourier(fy[0], fy[1], "t")
            return f"x(t) = {x_expr}  ;  y(t) = {y_expr}"

        cp = fitted_result.get("control_points", [])
        return f"Parametric Spline ({len(cp)} control points)"

    except Exception:
        return "Complex Curve"


def fit_all_contours(contours, smoothing=5, max_contours=50):
    results = []
    for i, contour in enumerate(contours):
        if i >= max_contours:
            break
        try:
            result = fit_contour_spline(contour, smoothing)
            result["index"] = i
            result["num_points"] = len(contour)
            result["expression"] = generate_expression(contour, result)
            results.append(result)
        except Exception:
            continue
    return results
