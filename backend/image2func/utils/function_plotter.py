import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_functions(fitted_results, width=10, height=8, show_points=True):
    fig, ax = plt.subplots(figsize=(width, height))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, result in enumerate(fitted_results):
        x = np.array(result["x"])
        y = np.array(result["y"])
        color = colors[i % len(colors)]
        ax.plot(x, y, color=color, linewidth=1.5, label=f"Curve {result.get('index', i)}")

    all_x = []
    all_y = []
    for r in fitted_results:
        all_x.extend(r["x"])
        all_y.extend(r["y"])

    if all_x and all_y:
        margin_x = (max(all_x) - min(all_x)) * 0.05 if max(all_x) != min(all_x) else 10
        margin_y = (max(all_y) - min(all_y)) * 0.05 if max(all_y) != min(all_y) else 10
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(max(all_y) + margin_y, min(all_y) - margin_y)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title("Function Plot from Image Contours")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
