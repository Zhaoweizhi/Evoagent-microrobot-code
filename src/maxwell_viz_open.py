"""
Maxwell 可视化脚本 — 使用最优 E-I 致动器设计参数建模、求解并保持 GUI 打开。
数据来源: AgenticOPT_20260209-235133_EI_RAG_RL.csv (iteration 19, fitness=-17.41)
"""

import os
import sys

os.environ["AEDT_NON_GRAPHICAL"] = "0"
os.environ["AEDT_CLOSE_ON_EXIT"] = "0"
os.environ["AEDT_NEW_DESKTOP"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxwell_pyaedt_run import (
    AEDT_VERSION,
    MaxwellSimulationConfig,
    _env_bool,
    compute_average_B,
    compute_mean_B_on_plane,
)
from pyaedt import Desktop, Maxwell3d

OUTPUT_DIR = os.environ.get(
    "MAXWELL_VIS_OUTPUT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "maxwell_outputs", "visualization")),
)
PROJECT_NAME = "EI_Actuator_Viz"
DESIGN_NAME = "BestDesign_Iter19"
SETUP_NAME = "Setup1"

config = MaxwellSimulationConfig(
    output_dir=OUTPUT_DIR,
    calc_fld_path=os.path.join(OUTPUT_DIR, "11.fld"),
    bsat_fld_path=os.path.join(OUTPUT_DIR, "Bsat_max.fld"),
    bsat_ta_fld_path=os.path.join(OUTPUT_DIR, "Bsat_ta.fld"),
    bsat_tb_fld_path=os.path.join(OUTPUT_DIR, "Bsat_tb.fld"),
    lm=4.72,
    ta=0.64,
    tb=1.0496,
    tm=0.5,
    dg=0.63,
    wa=2.0,
)


def build_and_solve():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    desktop = Desktop(
        version=AEDT_VERSION,
        non_graphical=False,
        new_desktop=True,
        close_on_exit=False,
    )

    app = Maxwell3d(
        projectname=PROJECT_NAME,
        designname=DESIGN_NAME,
        solution_type="Magnetostatic",
    )
    app.modeler.model_units = "mm"

    # ── 材料定义 ──────────────────────────────────────────────
    oDefMgr = app.oproject.GetDefinitionManager()

    _ndfe35_base = [
        "CoordinateSystemType:=", "Cartesian",
        "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic", "Thermal", "Structural"]],
        ["NAME:AttachedData",
         ["NAME:MatAppearanceData", "property_data:=", "appearance_data",
          "Red:=", 204, "Green:=", 204, "Blue:=", 204]],
        "permittivity:=", "1",
        "permeability:=", "1.0997785406",
        "conductivity:=", "625000",
        "dielectric_loss_tangent:=", "0",
        "magnetic_loss_tangent:=", "0",
    ]
    _ndfe35_tail = [
        "thermal_conductivity:=", "0",
        "saturation_mag:=", "0gauss",
        "lande_g_factor:=", "2",
        "delta_H:=", "0Oe",
        "mass_density:=", "7400",
        "youngs_modulus:=", "147000000000",
        ["NAME:thermal_expansion_coefficient",
         "property_type:=", "AnisoProperty", "unit:=", "",
         "component1:=", "3e-06", "component2:=", "-5e-06", "component3:=", "-5e-06"],
    ]

    def _coercivity_block(dir_z: str):
        return [
            "NAME:magnetic_coercivity",
            "property_type:=", "VectorProperty",
            "Magnitude:=", "-890000A_per_meter",
            "DirComp1:=", "0", "DirComp2:=", "0", "DirComp3:=", dir_z,
        ]

    oDefMgr.EditMaterial("NdFe35",
        ["NAME:NdFe35"] + _ndfe35_base + [_coercivity_block("1")] + _ndfe35_tail)
    oDefMgr.AddMaterial(
        ["NAME:NdFe35 - Copy"] + _ndfe35_base + [_coercivity_block("-1")] + _ndfe35_tail)

    # ── 几何建模 ──────────────────────────────────────────────
    seg_types = ["Line"] * (len(config.polyline1_points) - 1)
    app.modeler.create_polyline(
        points=config.polyline1_points,
        segment_type=seg_types,
        cover_surface=True, close_surface=True,
        name="Polyline1", material="vacuum",
    )
    app.modeler.create_rectangle(
        origin=config.rect1_origin, sizes=config.rect1_sizes,
        orientation="YZ", name="Rectangle1", material="vacuum",
    )
    app.modeler.create_rectangle(
        origin=config.rect2_origin, sizes=config.rect2_sizes,
        orientation="YZ", name="Rectangle2", material="vacuum",
    )
    app.modeler.create_rectangle(
        origin=config.rect3_origin, sizes=config.rect3_sizes,
        orientation="YZ", name="Rectangle3", material="vacuum",
    )

    for obj in ["Polyline1", "Rectangle1", "Rectangle2", "Rectangle3"]:
        if obj in app.modeler.object_names:
            app.modeler.sweep_along_vector(assignment=obj, sweep_vector=config.sweep_vector)

    app.assign_material("Polyline1", "DW310_35")
    app.assign_material("Rectangle1", "DW310_35")
    app.assign_material("Rectangle2", "NdFe35")
    app.assign_material("Rectangle3", "NdFe35 - Copy")

    # 观察体积 Box1（气隙区域）
    box1 = app.modeler.create_box(
        position=config.box1_position,
        dimensions_list=config.box1_size,
        name="Box1", material="vacuum",
    )
    try:
        box1.solve_inside = False
        box1.model = False
    except Exception:
        pass

    # 磁饱和检测面
    z_tb_start = config.ta + config.tm + config.dg
    z_ta_start = z_tb_start + config.tb + config.tm + config.dg
    mm = config._mm

    for pname, z0, sz_z in [
        ("BsatPlane_ta", z_ta_start, config.ta),
        ("BsatPlane_tb", z_tb_start, config.tb),
    ]:
        try:
            if pname in app.modeler.object_names:
                app.modeler.delete(pname)
            p = app.modeler.create_rectangle(
                origin=[mm(0), mm(config.ta), mm(z0)],
                sizes=[mm(sz_z), mm(config.wa)],
                orientation="XZ", name=pname, material="vacuum",
            )
            try:
                p.model = False
            except Exception:
                pass
        except Exception as e:
            print(f"创建 {pname} 失败: {e!r}")

    # ── 可视化辅助截面（YZ 中截面） ────────────────────────────
    # 在 X = wa/2 处放一个 YZ 截面矩形，方便后续画 B-field 云图
    z_total = 2 * config.ta + config.tb + 2 * config.tm + 2 * config.dg
    try:
        if "CrossSection_YZ" in app.modeler.object_names:
            app.modeler.delete("CrossSection_YZ")
        cs = app.modeler.create_rectangle(
            origin=[mm(config.wa / 2), mm(0), mm(0)],
            sizes=[mm(config.la), mm(z_total)],
            orientation="YZ", name="CrossSection_YZ", material="vacuum",
        )
        try:
            cs.model = False
        except Exception:
            pass
    except Exception as e:
        print(f"创建 CrossSection_YZ 失败: {e!r}")

    # 在 Y = la/2 处放一个 XZ 截面（纵截面）
    try:
        if "CrossSection_XZ" in app.modeler.object_names:
            app.modeler.delete("CrossSection_XZ")
        cs2 = app.modeler.create_rectangle(
            origin=[mm(0), mm(config.la / 2), mm(0)],
            sizes=[mm(config.wa), mm(z_total)],
            orientation="XZ", name="CrossSection_XZ", material="vacuum",
        )
        try:
            cs2.model = False
        except Exception:
            pass
    except Exception as e:
        print(f"创建 CrossSection_XZ 失败: {e!r}")

    # ── Region & 网格 ─────────────────────────────────────────
    region = app.modeler.create_region(pad_percent=[40, 40, 40, 40, 40, 40])
    try:
        region_name = region if isinstance(region, str) else getattr(region, "name", "Region")
        omesh = app.odesign.GetModule("MeshSetup")
        omesh.AssignLengthOp([
            "NAME:Length1", "RefineInside:=", False, "Enabled:=", True,
            "Objects:=", [region_name],
            "RestrictElem:=", False, "NumMaxElem:=", "1000",
            "RestrictLength:=", True, "MaxLength:=", "1mm",
        ])
        mesh_targets = [n for n in ["Polyline1", "Rectangle1", "Rectangle2", "Rectangle3"]
                        if n in app.modeler.object_names]
        mesh_targets.append(region_name)
        omesh.AssignLengthOp([
            "NAME:Length2", "RefineInside:=", True, "Enabled:=", True,
            "Objects:=", mesh_targets,
            "RestrictElem:=", False, "NumMaxElem:=", "1000",
            "RestrictLength:=", True, "MaxLength:=", "1mm",
        ])
    except Exception as e:
        print(f"网格控制设置出错: {e!r}")

    # ── 求解设置（更精细，为可视化服务） ───────────────────────
    setup = app.create_setup(SETUP_NAME)
    setup.props["MaximumPasses"] = 10
    setup.props["MinimumPasses"] = 2
    setup.props["MinimumConvergedPasses"] = 1
    setup.props["PercentRefinement"] = 30
    setup.props["PercentError"] = 1
    setup.props["SolveFieldOnly"] = False
    setup.props["UseIterativeSolver"] = False
    setup.update()

    # ── 求解 ──────────────────────────────────────────────────
    print("正在求解（MaxPasses=10, Error=1%, 网格1mm）...")
    try:
        app.mesh.generate_mesh(SETUP_NAME)
    except Exception:
        pass

    oproject = app.oproject
    odesign = app.odesign
    oproject.Save()
    odesign.AnalyzeAllNominal()
    oproject.Save()
    app.save_project()

    # ── 结果提取 ──────────────────────────────────────────────
    try:
        avg_B = compute_average_B(app, config.calc_fld_path,
                                  setup_name=SETUP_NAME, volume_name="Box1")
        b_ta = compute_mean_B_on_plane(app, config.bsat_ta_fld_path,
                                       setup_name=SETUP_NAME, plane_name="BsatPlane_ta")
        b_tb = compute_mean_B_on_plane(app, config.bsat_tb_fld_path,
                                       setup_name=SETUP_NAME, plane_name="BsatPlane_tb")
        print(f"\n{'='*60}")
        print(f"  avg_B (气隙平均Bz)     = {avg_B:.4f} T")
        print(f"  B_mean_ta (上轭平均|B|) = {b_ta:.4f} T  {'[!] 接近饱和!' if b_ta > 1.9 else ''}")
        print(f"  B_mean_tb (中间块平均|B|) = {b_tb:.4f} T")
        print(f"{'='*60}")
    except Exception as e:
        print(f"结果提取失败: {e!r}")

    # ── 创建 FieldOverlay（自动添加场图到模型上） ──────────────
    try:
        ofields = odesign.GetModule("FieldsReporter")

        # 创建命名表达式: Mag_B
        ofields.CalcStack("clear")
        ofields.EnterQty("B")
        ofields.CalcOp("Mag")
        ofields.AddNamedExpression("Mag_B", "Fields")

        # 在 CrossSection_YZ 上创建 |B| 场云图
        ofields.CreateFieldPlot(
            [
                "NAME:Mag_B_YZ_Plot",
                "SolutionName:=", f"{SETUP_NAME} : LastAdaptive",
                "UserSpecifyName:=", 0,
                "UserSpecifyFolder:=", 0,
                "QuantityName:=", "Mag_B",
                "PlotFolder:=", "Field",
                "StreamlinePlot:=", False,
                "AdjaacencyOn:=", False,
                "FullModelOn:=", False,
                "IntrinsicVar:=", "",
                "PlotGeomInfo:=", [1, "Surface", "FacesList", 1,
                                   str(app.modeler.get_object_from_name("CrossSection_YZ").faces[0].id)],
                "FilterBoxes:=", [0],
                [
                    "NAME:PlotOnSurfaceSettings",
                    "Filled:=", True,
                    "IsoValType:=", "Fringe",
                    "SmoothShade:=", True,
                    "AddGrid:=", False,
                    "MapTransparency:=", True,
                    "Refinement:=", 0,
                    "Transparency:=", 0,
                    "SmoothingLevel:=", 0,
                    [
                        "NAME:Arrow3DSpacingSettings",
                        "ArrowUniform:=", True,
                        "ArrowSpacing:=", 0,
                        "MinArrowSpacing:=", 0,
                        "MaxArrowSpacing:=", 0,
                    ],
                    "GridColor:=", [255, 255, 255],
                ],
                "EnableGaussianSmoothing:=", False,
            ],
        )
        print("已自动创建 |B| 场云图 (CrossSection_YZ)。")
    except Exception as e:
        print(f"自动创建场图时出错（可在GUI中手动创建）: {e!r}")

    print("\n" + "="*60)
    print("Maxwell GUI 已保持打开，请进行可视化操作。")
    print("="*60)

    desktop.release_desktop(close_projects=False, close_desktop=False)


if __name__ == "__main__":
    print("="*60)
    print("  E-I 致动器 Maxwell 可视化建模")
    print("="*60)
    print(f"  lm  = {config.lm} mm    (永磁体长度)")
    print(f"  ta  = {config.ta} mm    (上下轭厚度)")
    print(f"  tb  = {config.tb:.4f} mm (中间块高度)")
    print(f"  tm  = {config.tm} mm    (永磁体厚度)")
    print(f"  dg  = {config.dg} mm    (气隙)")
    print(f"  wa  = {config.wa} mm    (扫掠宽度)")
    print(f"  la  = {config.la:.2f} mm   (总长)")
    z_total = 2*config.ta + config.tb + 2*config.tm + 2*config.dg
    print(f"  ha  = {z_total:.4f} mm (总高)")
    print("="*60)
    print()
    build_and_solve()
