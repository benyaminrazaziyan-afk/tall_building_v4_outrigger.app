
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:
    class _StreamlitStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop
    st = _StreamlitStub()

# Local framework import
from outrigger_mdofframework import (
    Material,
    CoreWallSection,
    ColumnSection,
    OutriggerLevel,
    TowerInput,
    analyze_tower,
    redesign_with_outriggers,
    root_outrigger_study,
    plot_plan,
    plot_elevation,
    plot_modes,
    plot_design_history,
    example_input,
)

AUTHOR_NAME = "Benyamin"
APP_VERSION = "v5.0-Integrated-MDOF"

st.set_page_config(
    page_title="Tall Building Integrated MDOF + Outrigger",
    layout="wide",
    initial_sidebar_state="expanded",
)


def streamlit_input_panel() -> tuple[TowerInput, str]:
    st.markdown("### Geometry")
    c1, c2 = st.columns(2)
    with c1:
        n_story = st.number_input("Above-grade stories", min_value=2, max_value=120, value=60, step=1)
        story_height = st.number_input("Story height (m)", min_value=2.5, max_value=6.0, value=3.2)
        plan_x = st.number_input("Plan X (m)", min_value=20.0, max_value=300.0, value=80.0)
        n_perim_x_face = st.number_input("Perimeter columns on X face", min_value=2, max_value=30, value=7, step=1)
        floor_mass = st.number_input("Typical floor mass (kg)", min_value=1e5, max_value=5e6, value=9.0e5, step=1e4, format="%.0f")
    with c2:
        plan_y = st.number_input("Plan Y (m)", min_value=20.0, max_value=300.0, value=80.0)
        n_perim_y_face = st.number_input("Perimeter columns on Y face", min_value=2, max_value=30, value=7, step=1)
        axis = st.selectbox("Analysis axis", ["x", "y"], index=0)
        drift_den = st.number_input("Drift denominator", min_value=200.0, max_value=2000.0, value=500.0)

    st.markdown("### Core walls")
    c3, c4, c5 = st.columns(3)
    with c3:
        core_t = st.number_input("Core wall thickness (m)", min_value=0.20, max_value=1.50, value=0.60, step=0.05)
        core_lx = st.number_input("Core dimension X (m)", min_value=6.0, max_value=40.0, value=18.0, step=0.5)
    with c4:
        core_ly = st.number_input("Core dimension Y (m)", min_value=6.0, max_value=40.0, value=18.0, step=0.5)
        core_nx = st.number_input("Number of X core webs", min_value=1, max_value=6, value=2, step=1)
    with c5:
        core_ny = st.number_input("Number of Y core webs", min_value=1, max_value=6, value=2, step=1)
        core_cr = st.number_input("Core cracked factor", min_value=0.10, max_value=1.00, value=0.45, step=0.05)

    st.markdown("### Columns")
    c6, c7 = st.columns(2)
    with c6:
        perim_b = st.number_input("Perimeter column b (m)", min_value=0.40, max_value=2.50, value=0.95, step=0.05)
        perim_h = st.number_input("Perimeter column h (m)", min_value=0.40, max_value=2.50, value=0.95, step=0.05)
        perim_cr = st.number_input("Perimeter cracked factor", min_value=0.10, max_value=1.00, value=0.70, step=0.05)
    with c7:
        corner_b = st.number_input("Corner column b (m)", min_value=0.40, max_value=2.50, value=1.10, step=0.05)
        corner_h = st.number_input("Corner column h (m)", min_value=0.40, max_value=2.50, value=1.10, step=0.05)
        corner_cr = st.number_input("Corner cracked factor", min_value=0.10, max_value=1.00, value=0.70, step=0.05)

    st.markdown("### Materials")
    c8, c9 = st.columns(2)
    with c8:
        Ec = st.number_input("Concrete E (Pa)", min_value=20e9, max_value=60e9, value=32e9, step=1e9, format="%.0f")
        fy_col = st.number_input("Column fy (Pa)", min_value=200e6, max_value=700e6, value=420e6, step=10e6, format="%.0f")
    with c9:
        fy_wall = st.number_input("Wall strength proxy fy (Pa)", min_value=20e6, max_value=100e6, value=40e6, step=1e6, format="%.0f")

    st.markdown("### Outriggers")
    c10, c11, c12 = st.columns(3)
    with c10:
        n_or = st.number_input("Number of outriggers", min_value=0, max_value=5, value=0, step=1)
    with c11:
        truss_depth = st.number_input("Truss depth (m)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        chord_area = st.number_input("Chord area (m²)", min_value=0.01, max_value=0.50, value=0.08, step=0.01, format="%.3f")
    with c12:
        diag_area = st.number_input("Diagonal area (m²)", min_value=0.01, max_value=0.50, value=0.04, step=0.01, format="%.3f")
        Es_truss = st.number_input("Truss steel E (Pa)", min_value=100e9, max_value=250e9, value=200e9, step=10e9, format="%.0f")

    ou_story_levels: List[int] = []
    if int(n_or) > 0:
        st.markdown("**Outrigger story levels**")
        cols = st.columns(min(3, int(n_or)))
        defaults = [15, 30, 45, 50, 55]
        for i in range(int(n_or)):
            with cols[i % len(cols)]:
                lev = st.number_input(
                    f"Outrigger {i+1} story",
                    min_value=1,
                    max_value=int(n_story),
                    value=min(defaults[i], int(n_story)),
                    step=1,
                    key=f"or_story_{i}"
                )
                ou_story_levels.append(int(lev))

    core = CoreWallSection(
        thickness=float(core_t),
        length_x=float(core_lx),
        length_y=float(core_ly),
        n_web_x=int(core_nx),
        n_web_y=int(core_ny),
        cracked_factor=float(core_cr),
    )
    perim = ColumnSection(b=float(perim_b), h=float(perim_h), cracked_factor=float(perim_cr))
    corner = ColumnSection(b=float(corner_b), h=float(corner_h), cracked_factor=float(corner_cr))

    outriggers = [
        OutriggerLevel(
            story=s,
            axis=str(axis),
            truss_depth=float(truss_depth),
            chord_area=float(chord_area),
            diagonal_area=float(diag_area),
            truss_E=float(Es_truss),
        )
        for s in ou_story_levels
    ]

    inp = TowerInput(
        n_story=int(n_story),
        story_height=float(story_height),
        plan_x=float(plan_x),
        plan_y=float(plan_y),
        floor_mass=float(floor_mass),
        core=core,
        perimeter_column=perim,
        corner_column=corner,
        n_perimeter_columns_x_face=int(n_perim_x_face),
        n_perimeter_columns_y_face=int(n_perim_y_face),
        outriggers=outriggers,
        concrete_E=float(Ec),
        drift_limit_ratio=1.0 / float(drift_den),
        column_material=Material(E=float(Ec), fy=float(fy_col)),
        wall_material=Material(E=float(Ec), fy=float(fy_wall)),
    )
    return inp, str(axis)


def build_report(inp: TowerInput, axis: str, designed_model: TowerInput, final_result, hist_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("TALL BUILDING INTEGRATED MDOF + OUTRIGGER REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("INPUT MODEL")
    lines.append("-" * 78)
    lines.append(f"Stories                         = {inp.n_story}")
    lines.append(f"Story height                    = {inp.story_height:.3f} m")
    lines.append(f"Total height                    = {inp.height():.3f} m")
    lines.append(f"Plan                            = {inp.plan_x:.3f} x {inp.plan_y:.3f} m")
    lines.append(f"Analysis axis                   = {axis.upper()}")
    lines.append(f"Typical floor mass              = {inp.floor_mass:,.0f} kg")
    lines.append("")
    lines.append("FINAL DESIGNED SECTIONS")
    lines.append("-" * 78)
    lines.append(f"Core thickness                  = {designed_model.core.thickness:.3f} m")
    lines.append(f"Core dimensions                 = {designed_model.core.length_x:.3f} x {designed_model.core.length_y:.3f} m")
    lines.append(f"Perimeter column                = {designed_model.perimeter_column.b:.3f} x {designed_model.perimeter_column.h:.3f} m")
    lines.append(f"Corner column                   = {designed_model.corner_column.b:.3f} x {designed_model.corner_column.h:.3f} m")
    lines.append("")
    lines.append("DYNAMIC RESPONSE")
    lines.append("-" * 78)
    for i, T in enumerate(final_result.periods[:5], start=1):
        lines.append(f"Mode {i} period                 = {T:.4f} s")
    lines.append(f"Max drift ratio (unit profile)  = {final_result.max_drift_ratio_unit_profile:.6f}")
    lines.append("")
    if final_result.outriggers:
        lines.append("OUTRIGGER MECHANICS")
        lines.append("-" * 78)
        for mech in final_result.outriggers:
            lines.append(f"Story {mech.story:>3} | axis={mech.axis.upper()} | arm={mech.arm:.3f} m | "
                         f"k_truss={mech.k_truss_tip:,.3e} N/m | k_col={mech.k_column_tip:,.3e} N/m | "
                         f"k_story={mech.k_story_equiv:,.3e} N/m")
        lines.append("")
    if not hist_df.empty:
        lines.append("ITERATION HISTORY")
        lines.append("-" * 78)
        lines.append(hist_df.to_string(index=False))
        lines.append("")
    return "\n".join(lines)



def _candidate_stories(inp: TowerInput) -> list[int]:
    defaults = sorted(set([
        max(2, round(0.25 * inp.n_story)),
        max(3, round(0.50 * inp.n_story)),
        max(4, round(0.75 * inp.n_story)),
    ]))
    return defaults


def replace_for_study(inp: TowerInput) -> TowerInput:
    # same geometry/materials, no outriggers; study will add them
    return TowerInput(
        n_story=inp.n_story,
        story_height=inp.story_height,
        plan_x=inp.plan_x,
        plan_y=inp.plan_y,
        floor_mass=inp.floor_mass,
        core=inp.core,
        perimeter_column=inp.perimeter_column,
        corner_column=inp.corner_column,
        n_perimeter_columns_x_face=inp.n_perimeter_columns_x_face,
        n_perimeter_columns_y_face=inp.n_perimeter_columns_y_face,
        outriggers=[],
        concrete_E=inp.concrete_E,
        drift_limit_ratio=inp.drift_limit_ratio,
        column_material=inp.column_material,
        wall_material=inp.wall_material,
    )


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 0.7rem; padding-bottom: 0.7rem; max-width: 100%;}
    .stButton button {width: 100%; font-weight: 700; height: 3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tall Building Integrated MDOF + Outrigger")
st.caption(f"Prepared by {AUTHOR_NAME} | {APP_VERSION}")
st.info(
    "This version keeps the Streamlit UI, but the solver is driven by the new "
    "MDOF outrigger framework. The outrigger is assembled at its real level and "
    "feeds back into section re-sizing."
)

if "designed_model" not in st.session_state:
    st.session_state.designed_model = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "study_df" not in st.session_state:
    st.session_state.study_df = None
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

left_col, right_col = st.columns([1.0, 2.2], gap="medium")

with left_col:
    inp, axis = streamlit_input_panel()
    b1, b2 = st.columns(2)

    with b1:
        if st.button("ANALYZE / REDESIGN"):
            try:
                with st.spinner("Running integrated MDOF redesign..."):
                    model, result, hist_df = redesign_with_outriggers(inp, axis=axis, max_iter=6, verbose=False)
                    study_df = root_outrigger_study(
                        replace_for_study(inp),
                        candidate_stories=_candidate_stories(inp),
                        counts=(0, 1, 2, 3),
                        axis=axis,
                        redesign=True,
                    )
                    st.session_state.designed_model = model
                    st.session_state.analysis_result = result
                    st.session_state.history_df = hist_df
                    st.session_state.study_df = study_df
                    st.session_state.report_text = build_report(inp, axis, model, result, hist_df)
                st.success("Integrated analysis completed.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    with b2:
        if st.session_state.report_text:
            st.download_button(
                "SAVE REPORT",
                data=st.session_state.report_text.encode("utf-8"),
                file_name="integrated_mdof_outrigger_report.txt",
                mime="text/plain",
            )
        else:
            st.button("SAVE REPORT", disabled=True)

with right_col:
    if st.session_state.analysis_result is None:
        st.info("Run ANALYZE / REDESIGN to generate the new integrated results.")
    else:
        model = st.session_state.designed_model
        result = st.session_state.analysis_result
        hist_df = st.session_state.history_df
        study_df = st.session_state.study_df

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T1 (s)", f"{result.periods[0]:.3f}")
        c2.metric("T2 (s)", f"{result.periods[1]:.3f}" if len(result.periods) > 1 else "-")
        c3.metric("Max drift ratio", f"{result.max_drift_ratio_unit_profile:.5f}")
        c4.metric("Outriggers", len(result.outriggers))

        d1, d2, d3 = st.columns(3)
        d1.metric("Perim. column", f"{model.perimeter_column.b:.2f} x {model.perimeter_column.h:.2f} m")
        d2.metric("Corner column", f"{model.corner_column.b:.2f} x {model.corner_column.h:.2f} m")
        d3.metric("Core thickness", f"{model.core.thickness:.2f} m")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Plan / Elevation", "Modes", "Design History", "0-3 Outrigger Study", "Report"]
        )

        with tab1:
            a, b = st.columns(2)
            with a:
                st.pyplot(plot_plan(model, axis=axis), use_container_width=True)
            with b:
                st.pyplot(plot_elevation(model), use_container_width=True)

        with tab2:
            st.pyplot(plot_modes(result, n_modes=5), use_container_width=True)

        with tab3:
            if hist_df is not None and not hist_df.empty:
                st.pyplot(plot_design_history(hist_df), use_container_width=True)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.info("No iteration history available.")

        with tab4:
            if study_df is not None and not study_df.empty:
                st.dataframe(study_df, use_container_width=True, hide_index=True)
            else:
                st.info("Study table not available.")

        with tab5:
            st.text_area("", st.session_state.report_text, height=520, label_visibility="collapsed")
