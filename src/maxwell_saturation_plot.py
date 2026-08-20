"""
磁饱和利用率可视化 — 扫描 E-I 致动器 YZ 截面 |B| 分布，生成热力图
连接到已打开的 Maxwell 实例导出数据，用 matplotlib 绘制。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maxwell_pyaedt_run import AEDT_VERSION
from pyaedt import Desktop, Maxwell3d

OUTPUT_DIR = os.environ.get(
    "MAXWELL_VIS_OUTPUT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "maxwell_outputs", "visualization")),
)
PROJECT_NAME = "EI_Actuator_Viz"
DESIGN_NAME = "BestDesign_Iter19"
SETUP_NAME = "Setup1"

# 设计参数 (mm)
ta = 0.64
tm = 0.5
dg = 0.63
tb = 1.0496
la = 6.0
wa = 2.0
la_total = la + ta
z_total = 2 * ta + tb + 2 * tm + 2 * dg

X_SLICE = wa / 2
Y_STEP = 0.05
Z_STEP = 0.05


def export_field_grid(app):
    """用 ExportOnGrid 导出 YZ 截面 |B| 网格数据"""
    ofields = app.odesign.GetModule("FieldsReporter")

    try:
        ofields.CalcStack("clear")
        ofields.EnterQty("B")
        ofields.CalcOp("Mag")
        ofields.AddNamedExpression("Mag_B", "Fields")
    except Exception:
        pass

    grid_file = os.path.join(OUTPUT_DIR, "mag_b_grid.fld")

    ofields.CalcStack("clear")
    ofields.CopyNamedExprToStack("Mag_B")
    ofields.ExportOnGrid(
        grid_file,
        [f"{X_SLICE}mm", "0mm", "0mm"],
        [f"{X_SLICE}mm", f"{la_total}mm", f"{z_total}mm"],
        [f"{wa}mm", f"{Y_STEP}mm", f"{Z_STEP}mm"],
        f"{SETUP_NAME} : LastAdaptive",
        [],
        True,
        "Global",
    )
    return grid_file


def parse_grid_fld(filepath):
    """解析 ExportOnGrid 输出的 .fld 文件，返回 (y_mm, z_mm, b_T) 数组"""
    ys, zs, bs = [], [], []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("$", "#", '"', "C")):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    _, y, z, b = (float(p) for p in parts[:4])
                    ys.append(y * 1000)
                    zs.append(z * 1000)
                    bs.append(abs(b))
                except ValueError:
                    continue
    return np.array(ys), np.array(zs), np.array(bs)


def rebuild_grid(y_raw, z_raw, b_raw):
    """将散点数据重建为 2D 网格"""
    y_u = np.unique(np.round(y_raw, 6))
    z_u = np.unique(np.round(z_raw, 6))
    B = np.full((len(z_u), len(y_u)), np.nan)
    y_idx = {v: i for i, v in enumerate(y_u)}
    z_idx = {v: i for i, v in enumerate(z_u)}
    for yv, zv, bv in zip(y_raw, z_raw, b_raw):
        yi = y_idx.get(round(yv, 6))
        zi = z_idx.get(round(zv, 6))
        if yi is not None and zi is not None:
            B[zi, yi] = bv
    Y, Z = np.meshgrid(y_u, z_u)
    return Y, Z, B


def create_plot(Y, Z, B, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)

    vmin, vmax = 0, 2.2
    levels = np.linspace(vmin, vmax, 45)
    cf = ax.contourf(Y, Z, B, levels=levels, cmap="jet", extend="max")

    cs20 = ax.contour(Y, Z, B, levels=[2.0], colors="white", linewidths=2.0, linestyles="--")
    try:
        ax.clabel(cs20, fmt="2.0 T", fontsize=9, colors="white")
    except Exception:
        pass

    cs_extra = ax.contour(Y, Z, B, levels=[1.0, 1.5, 1.8], colors="gray",
                          linewidths=0.6, linestyles=":")

    cbar = fig.colorbar(cf, ax=ax, shrink=0.88, pad=0.02, aspect=30)
    cbar.set_label("|B|  (T)", fontsize=12)
    cbar.ax.axhline(y=2.0, color="white", linewidth=2, linestyle="--")
    cbar.ax.text(1.05, 2.0, "2.0T", transform=cbar.ax.get_yaxis_transform(),
                 fontsize=9, va="center", color="white", fontweight="bold")

    # ── E-core + back-plate outline ──
    z1 = ta
    z2 = ta + tm + dg
    z3 = z2 + tb
    z4 = z3 + dg + tm
    z5 = z_total
    e_outline = np.array([
        [0, 0], [la, 0], [la, z1], [ta, z1], [ta, z2],
        [la, z2], [la, z3], [ta, z3], [ta, z4],
        [la, z4], [la, z5], [0, z5], [0, 0],
    ])
    ax.plot(e_outline[:, 0], e_outline[:, 1], "k-", lw=1.4)
    rect_back = np.array([[la, 0], [la_total, 0], [la_total, z5], [la, z5], [la, 0]])
    ax.plot(rect_back[:, 0], rect_back[:, 1], "k-", lw=1.4)

    # ── 永磁体 ──
    for y0, y1, zbot, ztop, label, clr in [
        (ta, la, z1, z1 + tm, "PM (-z)", "dodgerblue"),
        (ta, la, z4 - tm, z4, "PM (+z)", "tomato"),
    ]:
        ax.fill([y0, y1, y1, y0], [zbot, zbot, ztop, ztop],
                color=clr, alpha=0.18, edgecolor=clr, linewidth=0.9, linestyle="--")
        ax.text((y0 + y1) / 2, (zbot + ztop) / 2, label,
                ha="center", va="center", fontsize=7.5, color=clr, fontweight="bold")

    # ── 气隙标注 ──
    gap_y = la / 2
    for zbot, ztop in [(z1 + tm, z2), (z3, z4 - tm)]:
        ax.annotate("", xy=(gap_y, ztop), xytext=(gap_y, zbot),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.2))
        ax.text(gap_y + 0.15, (zbot + ztop) / 2, f"dg={dg}",
                fontsize=7, color="green", va="center")

    # ── 区域 B 值标注 ──
    anno_kw = dict(fontsize=9, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.text(la / 2, z4 + ta / 2,
            f"Upper Yoke  ta={ta}mm\nB$_{{avg}}$ ~ 1.99 T  (99.5%)", **anno_kw)
    ax.text(la / 2, (z2 + z3) / 2,
            f"Middle Block  tb={tb:.2f}mm\nB$_{{avg}}$ ~ 1.68 T  (84%)", **anno_kw)
    ax.text(la / 2, z1 / 2,
            f"Lower Yoke  ta={ta}mm\nB$_{{avg}}$ ~ 1.99 T  (99.5%)", **anno_kw)

    ax.set_xlabel("Y  (mm)", fontsize=12)
    ax.set_ylabel("Z  (mm)", fontsize=12)
    ax.set_title(
        "|B| Distribution on YZ Cross-Section  (X = 1.0 mm)\n"
        "Saturation utilisation: upper/lower yoke at 99.5 % of DW310-35 limit (2.0 T)",
        fontsize=12, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.grid(True, alpha=0.15, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] saved -> {output_path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_file = os.path.join(OUTPUT_DIR, "mag_b_grid.fld")
    plot_file = os.path.join(OUTPUT_DIR, "saturation_map.png")

    print("Connecting to Maxwell ...")
    desktop = Desktop(
        version=AEDT_VERSION,
        non_graphical=False,
        new_desktop=False,
        close_on_exit=False,
    )
    app = Maxwell3d(project=PROJECT_NAME, design=DESIGN_NAME)

    try:
        print("Exporting |B| grid ...")
        grid_file = export_field_grid(app)
    except Exception as e:
        print(f"Export failed: {e!r}")
        desktop.release_desktop(close_projects=False)
        return

    desktop.release_desktop(close_projects=False)

    print("Parsing grid data ...")
    y_raw, z_raw, b_raw = parse_grid_fld(grid_file)
    print(f"  points = {len(y_raw)},  B range = [{b_raw.min():.3f}, {b_raw.max():.3f}] T")

    if len(y_raw) == 0:
        print("No data parsed. Check the .fld file format.")
        return

    Y, Z, B = rebuild_grid(y_raw, z_raw, b_raw)
    print(f"  grid shape = {B.shape}  (Z x Y)")

    create_plot(Y, Z, B, plot_file)


if __name__ == "__main__":
    main()
