from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.linalg import eigh
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for generalized eigenvalue solution") from exc

G = 9.81


@dataclass
class Material:
    E: float
    fy: float
    density: float = 7850.0


@dataclass
class CoreWallSection:
    thickness: float
    length_x: float
    length_y: float
    n_web_x: int = 2
    n_web_y: int = 2
    cracked_factor: float = 0.45

    @property
    def area(self) -> float:
        return self.n_web_x * self.length_x * self.thickness + self.n_web_y * self.length_y * self.thickness

    def inertia_y(self) -> float:
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        I_from_ywalls = self.n_web_y * (t * ly**3 / 12.0)
        arm = max(0.25 * lx, t / 2.0)
        A_x = lx * t
        I_from_xwalls = self.n_web_x * (lx * t**3 / 12.0 + A_x * arm**2)
        return self.cracked_factor * (I_from_ywalls + I_from_xwalls)

    def inertia_x(self) -> float:
        t = self.thickness
        lx = self.length_x
        ly = self.length_y
        I_from_xwalls = self.n_web_x * (t * lx**3 / 12.0)
        arm = max(0.25 * ly, t / 2.0)
        A_y = ly * t
        I_from_ywalls = self.n_web_y * (ly * t**3 / 12.0 + A_y * arm**2)
        return self.cracked_factor * (I_from_xwalls + I_from_ywalls)


@dataclass
class ColumnSection:
    b: float
    h: float
    cracked_factor: float = 0.70

    @property
    def area(self) -> float:
        return self.b * self.h

    def inertia_x(self) -> float:
        return self.cracked_factor * self.b * self.h**3 / 12.0

    def inertia_y(self) -> float:
        return self.cracked_factor * self.h * self.b**3 / 12.0


@dataclass
class OutriggerLevel:
    story: int
    axis: str = "x"
    truss_depth: float = 4.0
    chord_area: float = 0.12
    diagonal_area: float = 0.06
    truss_E: float = 200e9

    def validate(self) -> None:
        if self.axis.lower() not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")
        if self.story < 1:
            raise ValueError("story must be >= 1")


@dataclass
class TowerInput:
    n_story: int
    story_height: float
    plan_x: float
    plan_y: float
    floor_mass: float
    core: CoreWallSection
    perimeter_column: ColumnSection
    corner_column: ColumnSection
    n_perimeter_columns_x_face: int
    n_perimeter_columns_y_face: int
    outriggers: List[OutriggerLevel] = field(default_factory=list)
    column_material: Material = field(default_factory=lambda: Material(E=32e9, fy=420e6))
    wall_material: Material = field(default_factory=lambda: Material(E=32e9, fy=40e6))
    design_base_shear_ratio: float = 0.02
    core_stiffness_modifier: float = 1.0
    column_stiffness_modifier: float = 1.0

    def height(self) -> float:
        return self.n_story * self.story_height

    def bay_arm(self, axis: str) -> float:
        if axis.lower() == "x":
            return max((self.plan_x - self.core.length_x) / 2.0, 0.5)
        return max((self.plan_y - self.core.length_y) / 2.0, 0.5)

    def n_engaged_columns_per_side(self, axis: str) -> int:
        return self.n_perimeter_columns_y_face if axis.lower() == "x" else self.n_perimeter_columns_x_face

    def validate(self) -> None:
        if self.n_story < 2:
            raise ValueError("n_story must be >= 2")
        if self.story_height <= 0 or self.floor_mass <= 0:
            raise ValueError("story_height and floor_mass must be positive")
        for ou in self.outriggers:
            ou.validate()
            if ou.story > self.n_story:
                raise ValueError(f"outrigger story {ou.story} > n_story {self.n_story}")


@dataclass
class OutriggerMechanics:
    story: int
    axis: str
    arm: float
    k_truss_tip: float
    k_column_tip: float
    k_tip_combined: float
    k_rot: float
    distributed_story_stiffness: float
    axial_force_per_side: float


@dataclass
class AnalysisResult:
    periods: np.ndarray
    frequencies: np.ndarray
    mode_shapes: np.ndarray
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    story_stiffness_base: np.ndarray
    story_stiffness_total: np.ndarray
    outriggers: List[OutriggerMechanics]
    floor_displacements: np.ndarray
    story_drifts: np.ndarray
    outrigger_axial_by_level: dict[int, float]
    max_drift_ratio: float
    base_shear: float
    input_model: TowerInput


@dataclass
class DesignIterationResult:
    iteration: int
    period_1: float
    perimeter_col_b: float
    perimeter_col_h: float
    corner_col_b: float
    corner_col_h: float
    core_thickness: float
    max_outrigger_axial_kN: float
    max_drift_ratio: float


def _ensure_vector_mass(inp: TowerInput) -> np.ndarray:
    return np.full(inp.n_story, inp.floor_mass, dtype=float)


def _assemble_shear_building_K(story_k: np.ndarray) -> np.ndarray:
    n = len(story_k)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += story_k[i]
        else:
            K[i, i] += story_k[i]
            K[i, i - 1] -= story_k[i]
            K[i - 1, i] -= story_k[i]
            K[i - 1, i - 1] += story_k[i]
    return K


def _assemble_diagonal_mass(masses: np.ndarray) -> np.ndarray:
    return np.diag(masses)


def _story_stiffness_from_core(inp: TowerInput, axis: str) -> np.ndarray:
    axis = axis.lower()
    I = inp.core.inertia_y() if axis == "x" else inp.core.inertia_x()
    E = inp.wall_material.E
    H = inp.height()
    k_global = inp.core_stiffness_modifier * 3.0 * E * I / max(H**3, 1e-12)
    return np.full(inp.n_story, inp.n_story * k_global, dtype=float)


def _story_stiffness_from_columns(inp: TowerInput, axis: str) -> np.ndarray:
    axis = axis.lower()
    E = inp.column_material.E
    if axis == "x":
        I_per = inp.perimeter_column.inertia_y()
        I_cor = inp.corner_column.inertia_y()
    else:
        I_per = inp.perimeter_column.inertia_x()
        I_cor = inp.corner_column.inertia_x()
    n_per_cols = 2 * inp.n_engaged_columns_per_side(axis)
    n_corner = 4
    I_total = n_per_cols * I_per + n_corner * I_cor
    H = inp.height()
    k_global = inp.column_stiffness_modifier * 3.0 * E * I_total / max(H**3, 1e-12)
    return np.full(inp.n_story, inp.n_story * k_global, dtype=float)


def _engaged_column_axial_stiffness(inp: TowerInput, story: int, axis: str) -> float:
    E = inp.column_material.E
    L = max(story * inp.story_height, inp.story_height)
    n_side = inp.n_engaged_columns_per_side(axis)
    A_per = inp.perimeter_column.area
    A_cor = inp.corner_column.area
    A_side_total = n_side * A_per + 1.0 * A_cor
    return E * A_side_total / L


def _outrigger_truss_tip_stiffness(arm: float, truss_depth: float, chord_area: float, diagonal_area: float, truss_E: float) -> float:
    d = max(truss_depth, 0.25)
    b = max(arm, 0.25)
    Ld = sqrt(b**2 + d**2)
    Ieq = 0.5 * chord_area * d**2
    k_bend = 3.0 * truss_E * Ieq / max(b**3, 1e-12)
    k_diag = 2.0 * truss_E * diagonal_area * d**2 / max(Ld**3, 1e-12)
    return k_bend + k_diag


def outrigger_mechanics(inp: TowerInput, ou: OutriggerLevel) -> OutriggerMechanics:
    axis = ou.axis.lower()
    arm = inp.bay_arm(axis)
    k_truss_tip = _outrigger_truss_tip_stiffness(arm, ou.truss_depth, ou.chord_area, ou.diagonal_area, ou.truss_E)
    k_col_tip = _engaged_column_axial_stiffness(inp, ou.story, axis)
    k_tip = 1.0 / (1.0 / max(k_truss_tip, 1e-12) + 1.0 / max(k_col_tip, 1e-12))
    k_rot = 2.0 * k_tip * arm**2
    distributed_story_stiffness = k_rot / max(ou.story * inp.story_height**2, 1e-12)
    return OutriggerMechanics(
        story=ou.story,
        axis=axis,
        arm=arm,
        k_truss_tip=k_truss_tip,
        k_column_tip=k_col_tip,
        k_tip_combined=k_tip,
        k_rot=k_rot,
        distributed_story_stiffness=distributed_story_stiffness,
        axial_force_per_side=0.0,
    )


def _triangular_lateral_force(inp: TowerInput) -> tuple[np.ndarray, float]:
    masses = _ensure_vector_mass(inp)
    weights = masses * G
    Vb = inp.design_base_shear_ratio * weights.sum()
    z = np.arange(1, inp.n_story + 1, dtype=float) * inp.story_height
    profile = z / z.sum()
    return Vb * profile, Vb


def analyze_tower(inp: TowerInput, axis: str = "x") -> AnalysisResult:
    inp.validate()
    axis = axis.lower()
    m = _ensure_vector_mass(inp)
    k_core = _story_stiffness_from_core(inp, axis)
    k_cols = _story_stiffness_from_columns(inp, axis)
    k_total = k_core + k_cols

    ou_mech: List[OutriggerMechanics] = []
    for ou in inp.outriggers:
        if ou.axis.lower() == axis:
            mech = outrigger_mechanics(inp, ou)
            ou_mech.append(mech)
            k_total[:ou.story] += mech.distributed_story_stiffness

    K = _assemble_shear_building_K(k_total)
    M = _assemble_diagonal_mass(m)
    w2, phi = eigh(K, M)
    mask = w2 > 1e-9
    w = np.sqrt(w2[mask])
    phi = phi[:, mask]
    periods = 2.0 * pi / w
    freqs = w / (2.0 * pi)

    phi_n = phi.copy()
    for i in range(phi_n.shape[1]):
        scale = phi_n[-1, i] if abs(phi_n[-1, i]) > 1e-12 else np.max(np.abs(phi_n[:, i]))
        phi_n[:, i] /= scale
        if phi_n[-1, i] < 0:
            phi_n[:, i] *= -1.0

    f_lat, Vb = _triangular_lateral_force(inp)
    u = np.linalg.solve(K, f_lat)
    drift = np.diff(np.hstack(([0.0], u))) / inp.story_height
    max_drift = float(np.max(np.abs(drift)))

    ou_axial = {}
    updated = []
    for mech in ou_mech:
        j = mech.story - 1
        theta = u[j] / max(mech.story * inp.story_height, 1e-12)
        delta_tip = theta * mech.arm
        axial = mech.k_tip_combined * delta_tip
        ou_axial[j + 1] = axial
        updated.append(replace(mech, axial_force_per_side=axial))

    return AnalysisResult(
        periods=periods,
        frequencies=freqs,
        mode_shapes=phi_n,
        mass_matrix=M,
        stiffness_matrix=K,
        story_stiffness_base=k_core + k_cols,
        story_stiffness_total=k_total,
        outriggers=updated,
        floor_displacements=u,
        story_drifts=drift,
        outrigger_axial_by_level=ou_axial,
        max_drift_ratio=max_drift,
        base_shear=Vb,
        input_model=inp,
    )


def _update_perimeter_columns(inp: TowerInput, result: AnalysisResult, min_dim: float = 0.80, max_dim: float = 2.20) -> ColumnSection:
    Nmax = max(result.outrigger_axial_by_level.values(), default=0.0)
    if Nmax <= 1.0:
        return inp.perimeter_column
    fy = inp.column_material.fy
    demand_area = 1.15 * Nmax / max(0.40 * fy, 1e-12)
    target_a = max(inp.perimeter_column.area, demand_area)
    side = min(max_dim, max(min_dim, sqrt(target_a)))
    return ColumnSection(side, side, inp.perimeter_column.cracked_factor)


def _update_corner_columns(inp: TowerInput, perimeter_col: ColumnSection, min_dim: float = 0.90, max_dim: float = 2.40) -> ColumnSection:
    target_a = max(inp.corner_column.area, 1.15 * perimeter_col.area)
    side = min(max_dim, max(min_dim, sqrt(target_a)))
    return ColumnSection(side, side, inp.corner_column.cracked_factor)


def _update_core(inp: TowerInput, result: AnalysisResult, max_drift_ratio_target: float = 1 / 500.0) -> CoreWallSection:
    if result.max_drift_ratio <= max_drift_ratio_target:
        return inp.core
    ratio = min(max(result.max_drift_ratio / max(max_drift_ratio_target, 1e-12), 1.0), 1.20)
    new_t = min(inp.core.thickness * sqrt(ratio), 1.50)
    return replace(inp.core, thickness=new_t)


def redesign_with_outriggers(inp: TowerInput, axis: str = "x", max_iter: int = 6, verbose: bool = False):
    model = replace(inp)
    history = []
    for it in range(1, max_iter + 1):
        res = analyze_tower(model, axis)
        new_per = _update_perimeter_columns(model, res)
        new_cor = _update_corner_columns(model, new_per)
        tmp = replace(model, perimeter_column=new_per, corner_column=new_cor)
        res_tmp = analyze_tower(tmp, axis)
        new_core = _update_core(tmp, res_tmp)
        new_model = replace(tmp, core=new_core)

        history.append(DesignIterationResult(
            iteration=it,
            period_1=float(res.periods[0]),
            perimeter_col_b=model.perimeter_column.b,
            perimeter_col_h=model.perimeter_column.h,
            corner_col_b=model.corner_column.b,
            corner_col_h=model.corner_column.h,
            core_thickness=model.core.thickness,
            max_outrigger_axial_kN=max(res.outrigger_axial_by_level.values(), default=0.0) / 1e3,
            max_drift_ratio=res.max_drift_ratio,
        ))

        if verbose:
            print(
                f"it={it:02d}, T1={res.periods[0]:.3f} s, "
                f"PerCol={model.perimeter_column.b:.2f}x{model.perimeter_column.h:.2f} m, "
                f"Core t={model.core.thickness:.2f} m, "
                f"N_ou,max={max(res.outrigger_axial_by_level.values(), default=0.0)/1e3:.1f} kN"
            )

        diffs = [
            abs(new_model.perimeter_column.area - model.perimeter_column.area) / max(model.perimeter_column.area, 1e-12),
            abs(new_model.corner_column.area - model.corner_column.area) / max(model.corner_column.area, 1e-12),
            abs(new_model.core.thickness - model.core.thickness) / max(model.core.thickness, 1e-12),
        ]
        model = new_model
        if max(diffs) < 0.01:
            break

    final = analyze_tower(model, axis)
    hist_df = pd.DataFrame([x.__dict__ for x in history])
    return model, final, hist_df


def root_outrigger_study(base_inp: TowerInput, candidate_stories: Sequence[int], counts: Sequence[int] = (0, 1, 2, 3), axis: str = "x", redesign: bool = True) -> pd.DataFrame:
    rows = []
    template_ou = OutriggerLevel(story=1, axis=axis)
    for c in counts:
        levels = [replace(template_ou, story=int(s), axis=axis) for s in list(candidate_stories)[:c]]
        inp = replace(base_inp, outriggers=levels)
        if redesign:
            model, res, _ = redesign_with_outriggers(inp, axis=axis, max_iter=6, verbose=False)
        else:
            model, res = inp, analyze_tower(inp, axis=axis)
        rows.append({
            "n_outriggers": c,
            "stories": [ou.story for ou in model.outriggers],
            "T1_s": float(res.periods[0]),
            "roof_disp_mm": float(res.floor_displacements[-1] * 1e3),
            "max_drift_ratio": float(res.max_drift_ratio),
            "perimeter_col_m": f"{model.perimeter_column.b:.2f} x {model.perimeter_column.h:.2f}",
            "corner_col_m": f"{model.corner_column.b:.2f} x {model.corner_column.h:.2f}",
            "core_t_m": model.core.thickness,
            "max_outrigger_axial_kN": max(res.outrigger_axial_by_level.values(), default=0.0) / 1e3,
        })
    return pd.DataFrame(rows)


def _column_points(inp: TowerInput):
    xs = np.linspace(0.0, inp.plan_x, inp.n_perimeter_columns_x_face + 2)[1:-1]
    ys = np.linspace(0.0, inp.plan_y, inp.n_perimeter_columns_y_face + 2)[1:-1]
    pts = []
    for x in xs:
        pts.extend([(x, 0.0), (x, inp.plan_y)])
    for y in ys:
        pts.extend([(0.0, y), (inp.plan_x, y)])
    pts.extend([(0.0, 0.0), (inp.plan_x, 0.0), (0.0, inp.plan_y), (inp.plan_x, inp.plan_y)])
    return pts


def plot_plan(inp: TowerInput, axis: str = "x", figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, inp.plan_x, inp.plan_x, 0, 0], [0, 0, inp.plan_y, inp.plan_y, 0], lw=2)
    cx0 = 0.5 * (inp.plan_x - inp.core.length_x)
    cy0 = 0.5 * (inp.plan_y - inp.core.length_y)
    ax.add_patch(plt.Rectangle((cx0, cy0), inp.core.length_x, inp.core.length_y, fill=False, lw=2))
    pts = np.array(_column_points(inp))
    ax.scatter(pts[:, 0], pts[:, 1], s=30)

    if axis.lower() == "x":
        ymid = inp.plan_y / 2.0
        ax.plot([cx0, 0.0], [ymid, ymid], lw=3)
        ax.plot([cx0 + inp.core.length_x, inp.plan_x], [ymid, ymid], lw=3)
    else:
        xmid = inp.plan_x / 2.0
        ax.plot([xmid, xmid], [cy0, 0.0], lw=3)
        ax.plot([xmid, xmid], [cy0 + inp.core.length_y, inp.plan_y], lw=3)

    ax.set_aspect("equal")
    ax.set_title("Plan view: perimeter columns, core walls, and outrigger axis")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_elevation(inp: TowerInput, figsize=(7, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    H = inp.height()
    x_core_left, x_core_right = 0.35, 0.65
    x_col_left, x_col_right = 0.05, 0.95
    ax.plot([x_col_left, x_col_left], [0, H], lw=1.5)
    ax.plot([x_col_right, x_col_right], [0, H], lw=1.5)
    ax.plot([x_core_left, x_core_left], [0, H], lw=3)
    ax.plot([x_core_right, x_core_right], [0, H], lw=3)
    for i in range(inp.n_story + 1):
        z = i * inp.story_height
        ax.plot([x_col_left, x_col_right], [z, z], lw=0.35, alpha=0.35)
    for ou in inp.outriggers:
        z = ou.story * inp.story_height
        ax.plot([x_core_left, x_col_left], [z, z], lw=2.5)
        ax.plot([x_core_right, x_col_right], [z, z], lw=2.5)
        ax.text(1.0, z, f"Outrigger @ story {ou.story}", va="center")
    ax.set_title("Elevation view: core, perimeter columns, and outrigger levels")
    ax.set_xlabel("schematic width")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_modes(result: AnalysisResult, n_modes: int = 3):
    n_modes = min(n_modes, result.mode_shapes.shape[1])
    z = np.arange(1, result.input_model.n_story + 1) * result.input_model.story_height
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in range(n_modes):
        ax.plot(result.mode_shapes[:, i], z, label=f"Mode {i+1} (T={result.periods[i]:.3f} s)")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Normalized modal displacement")
    ax.set_ylabel("Height (m)")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_design_history(hist_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_df["iteration"], hist_df["period_1"], marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("T1 (s)")
    ax.set_title("Redesign history")
    ax.grid(True, alpha=0.25)
    return fig, ax


def example_input() -> TowerInput:
    return TowerInput(
        n_story=60,
        story_height=3.2,
        plan_x=80.0,
        plan_y=80.0,
        floor_mass=9.0e5,
        core=CoreWallSection(thickness=0.60, length_x=18.0, length_y=18.0, n_web_x=2, n_web_y=2, cracked_factor=0.45),
        perimeter_column=ColumnSection(b=0.95, h=0.95, cracked_factor=0.70),
        corner_column=ColumnSection(b=1.10, h=1.10, cracked_factor=0.70),
        n_perimeter_columns_x_face=7,
        n_perimeter_columns_y_face=7,
        outriggers=[],
        column_material=Material(E=32e9, fy=420e6),
        wall_material=Material(E=32e9, fy=40e6),
        design_base_shear_ratio=0.02,
    )


def example_usage():
    inp = example_input()
    print("=== Bare tower ===")
    bare = analyze_tower(inp, axis="x")
    print(f"T1 = {bare.periods[0]:.3f} s")
    print(f"Roof displacement = {bare.floor_displacements[-1]*1e3:.1f} mm")

    print("\n=== Tower with outriggers at 15, 30, 45 ===")
    inp2 = replace(inp, outriggers=[OutriggerLevel(15, "x"), OutriggerLevel(30, "x"), OutriggerLevel(45, "x")])
    model2, res2, hist = redesign_with_outriggers(inp2, axis="x", max_iter=6, verbose=True)
    print(f"Final T1 = {res2.periods[0]:.3f} s")
    print(f"Final perimeter column = {model2.perimeter_column.b:.2f} x {model2.perimeter_column.h:.2f} m")
    print(f"Final core thickness = {model2.core.thickness:.2f} m")
    print(f"Max outrigger axial = {max(res2.outrigger_axial_by_level.values(), default=0.0)/1e3:.1f} kN")

    study = root_outrigger_study(inp, candidate_stories=[15, 30, 45], counts=(0, 1, 2, 3), axis="x", redesign=True)
    print("\n=== Comparison study ===")
    print(study.to_string(index=False))
    return inp, bare, model2, res2, hist, study


if __name__ == "__main__":
    example_usage()
