# tall_building_v4_outrigger.app
https://tallbuildingv4outriggerapp-c2gaoqfca7di6woww8pvod.streamlit.app/
Tall Building Outrigger Predesign Framework — Version 4.4
Author: BENYAMIN RAZAZIYAN  
Version: 4.4-reviewed-preliminary-framework  
Project type: Preliminary structural design / computational framework  
Main file: `tower_outrigger_predesign_framework_v4_4_reviewed.py`
---
1. Project Objective
This project provides a preliminary computational framework for evaluating the lateral behavior of tall buildings with and without outrigger systems. The tool is intended to support early-stage structural comparison, parametric studies, and research-oriented investigation of how outrigger location, braced-span configuration, concrete strength, core-wall dimensions, perimeter walls, basement retaining walls, and seismic spectrum parameters influence global stiffness, modal period, drift, and preliminary member sizing.
The framework is not intended to replace a complete three-dimensional finite-element design model. Its main purpose is to create an engineering-based predesign model that can be compared and calibrated against professional software such as ETABS.
---
2. Engineering Concept
The model represents the building as a vertical flexural multi-degree-of-freedom system. Each floor level includes lateral displacement and rotation degrees of freedom. The stiffness matrix is assembled from the effective flexural stiffness of the structural system, including:
central reinforced-concrete core walls,
perimeter and side wall contribution,
column group stiffness,
optional basement retaining-wall contribution,
outrigger stiffness from actual braced bay panels,
collector/belt action at outrigger levels.
The outrigger system is treated as a braced-bay stiffness mechanism, not only as a visual line in the plan. The selected braced bays are used consistently in the plan view, stiffness calculation, and global structural model.
---
3. Main Capabilities
Preliminary sizing of core walls, columns, beams, slabs, and side/perimeter walls.
X- and Y-direction modal analysis.
First-mode period estimation using an assembled MDOF stiffness and mass model.
ASCE 7 style response-spectrum analysis input using either direct `SDS`, `SD1` or mapped values `SS`, `S1`, `Fa`, `Fv`.
Equivalent lateral force base-shear scaling.
Drift-controlled redesign loop.
Outrigger comparison with and without braced-bay action.
Plan visualization of grid, core, columns, perimeter walls, and outrigger bracing layout.
Story-by-story response tables.
Modal participation tables.
Preliminary material quantity estimation.
---
4. Required Python Packages
```bash
pip install numpy pandas matplotlib scipy streamlit
```
Recommended file structure:
```text
project-folder/
│
├── tower_outrigger_predesign_framework_v4_4_reviewed.py
├── README.md
└── requirements.txt
```
Optional `requirements.txt`:
```text
numpy
pandas
matplotlib
scipy
streamlit
```
---
5. How to Run
Option A — Streamlit
```bash
streamlit run tower_outrigger_predesign_framework_v4_4_reviewed.py
```
Option B — Python / Jupyter
```python
from tower_outrigger_predesign_framework_v4_4_reviewed import BuildingInput, run_design, summary_table

inp = BuildingInput()
res = run_design(inp)

summary_table(res)
```
---
6. Main Inputs and Their Engineering Role
6.1 Geometry Inputs
Input	Meaning	Engineering role
`n_story`	Number of stories above ground	Controls total height, mass distribution, period, drift demand
`n_basement`	Number of basement levels	Activates below-grade retaining-wall stiffness contribution
`story_height`	Typical story height	Controls total height and member stiffness length
`plan_x`, `plan_y`	Plan dimensions	Controls bay size, lever arm, floor area, mass, and outrigger arm
`n_bays_x`, `n_bays_y`	Number of grid bays	Defines column grid and possible braced bay locations
6.2 Material Inputs
Input	Meaning	Engineering role
`fck`	Concrete compressive strength	Affects preliminary section sizing and effective stiffness through material strength logic
`Ec`	Concrete elastic modulus	Directly affects global lateral stiffness and modal period
`fy`	Steel yield strength	Used for preliminary reinforcement/material parameters
Increasing concrete strength and elastic modulus should influence the preliminary sizes and the global stiffness. If a section reaches a minimum practical or code-based size, further strength increase may no longer reduce that section.
6.3 Load and Mass Inputs
Input	Meaning	Engineering role
`DL`	Dead load	Contributes to seismic mass and gravity sizing
`LL`	Live load	Partially included in seismic mass
`live_load_mass_factor`	Live load participation factor	Controls seismic mass contribution
`slab_finish_allowance`	Additional floor dead load	Increases mass and gravity demand
`facade_line_load`	Facade load along perimeter	Adds seismic mass and total weight
6.4 Outrigger Inputs
Input	Meaning	Engineering role
`outrigger_system`	None, tubular bracing, or belt truss	Controls whether outrigger action is activated
`outrigger_count`	Number of outrigger levels	Controls how many outrigger floors are active
`outrigger_story_levels`	Candidate outrigger stories	Defines outrigger vertical location
`braced_spans_x`, `braced_spans_y`	Number of braced spans	Controls total outrigger brace stiffness
`tubular_diameter_m`, `tubular_thickness_m`	Steel tube dimensions	Controls axial stiffness `EA/L` of each brace
`outrigger_connection_efficiency`	Connection/load-transfer efficiency	Reduces ideal stiffness to account for non-perfect connection
The outrigger stiffness follows the basic concept:
```math
K_{out} = \sum \frac{EA}{L}\cos^2\theta
```
The stiffness of one brace panel is multiplied by the number of active braces and braced spans. This stiffness is then transformed into the global lateral system through the outrigger lever arm and added to the global stiffness matrix.
6.5 Seismic Spectrum Inputs
Two spectrum input modes are supported:
Direct design spectrum input:
```text
SDS, SD1
```
Site-coefficient input:
```text
SS, S1, Fa, Fv
```
with:
```math
SDS = \frac{2}{3} F_a S_S
```
```math
SD1 = \frac{2}{3} F_v S_1
```
Other seismic parameters include:
Input	Meaning
`R`	Response modification factor
`Ie`	Importance factor
`Cd`	Deflection amplification factor
`TL`	Long-period transition period
`damping_ratio`	Modal damping ratio
---
7. Main Outputs
The program produces several engineering outputs:
7.1 Summary Table
Includes:
total height,
floor area,
total seismic weight,
total mass,
first modal period in X and Y,
modal mass participation,
base shear,
maximum drift ratio,
total concrete quantity,
total steel quantity,
active outrigger system.
7.2 Final Dimension Table
Reports representative section sizes for lower, middle, and upper zones:
core dimensions,
core wall thickness,
side/perimeter wall dimensions,
interior/perimeter/corner column sizes,
beam dimensions,
slab thickness.
7.3 Story Response Tables
For each story and direction:
floor force,
story shear,
overturning moment,
lateral displacement,
interstory drift,
drift ratio.
7.4 Modal Tables
For each mode:
period,
frequency,
participation factor,
effective modal mass,
cumulative modal mass.
7.5 Stiffness Tables
Includes story-by-story stiffness data:
effective `EI` in X and Y,
outrigger rotational stiffness,
outrigger lateral stiffness,
story mass,
concrete and steel quantities.
7.6 Plots
The framework can generate:
plan view,
mode shapes,
drift profile,
story shear profile,
stiffness profile,
comparative outrigger behavior.
---
8. What Was Corrected in Version 4.4
Version 4.4 focuses on making the model more suitable for research comparison and GitHub publication.
Main corrections include:
The author name is fixed as BENYAMIN RAZAZIYAN.
The seismic spectrum input was corrected to include `SS`, `S1`, `Fa`, and `Fv` in addition to direct `SDS` and `SD1`.
The outrigger system uses braced-bay stiffness instead of only a visual or arbitrary spring representation.
Braced-span count is connected to stiffness contribution.
Perimeter walls are included in the plan and stiffness model.
Basement retaining-wall stiffness is included as an approximate below-grade contribution when basement stories are defined.
Concrete strength and elastic modulus effects are linked to preliminary member sizing and global stiffness.
The code title and metadata were cleaned for repository presentation.
---
9. Engineering Limitations
This framework is a preliminary research and predesign tool. It has important limitations:
It is not a full 3D finite-element model.
It does not replace ETABS, SAP2000, SAFE, Abaqus, or code-based final design software.
Torsional irregularity is not fully modeled.
Diaphragm flexibility is simplified.
Soil-structure interaction is not explicitly modeled.
Basement retaining-wall stiffness is approximate.
Outrigger connections are represented by efficiency factors, not detailed joint models.
Member design is preliminary and does not replace reinforced-concrete design checks.
Shear-wall boundary elements, confinement, reinforcement detailing, and nonlinear behavior are not fully designed.
Wind load, construction sequence, creep, shrinkage, P-Delta effects, and second-order nonlinearities require separate advanced verification.
For journal publication, the results should be validated against a detailed ETABS or other finite-element model.
---
10. Recommended Validation Procedure Against ETABS
For research or Q1 journal-level use, the following validation path is recommended:
Build the same building geometry in ETABS.
Use the same story heights, plan dimensions, material properties, and mass source.
Define the same core walls, perimeter walls, columns, beams, and slabs.
Assign cracked stiffness modifiers consistent with the Python framework.
Model outrigger levels at the same story elevations.
Use the same number and location of braced bays.
Compare:
first modal period,
X/Y modal mass participation,
base shear,
maximum drift ratio,
drift profile shape,
story shear profile,
effect of removing/adding outriggers.
Calibrate connection efficiency and cracked stiffness factors if required.
The framework should be judged by trend accuracy and preliminary prediction capability, not by exact replacement of a full ETABS model.
---
11. Suggested Academic Description
This tool can be described in an article as:
> A preliminary computational framework was developed to evaluate the influence of outrigger-braced bay systems on the lateral response of tall buildings. The framework uses a flexural multi-degree-of-freedom stick model with translational and rotational degrees of freedom at each floor level. Core-wall, column-group, perimeter-wall, basement-wall, and outrigger braced-bay stiffness contributions are assembled into the global stiffness matrix. Modal response-spectrum analysis and drift-controlled preliminary redesign are implemented to compare structural behavior with and without outrigger action. The framework is intended for early-stage structural assessment and parametric comparison, and its results should be validated using detailed three-dimensional finite-element software such as ETABS.
---
12. Citation / Repository Note
If this repository is used in an academic study, cite the author and mention that the tool is a preliminary computational framework for tall-building outrigger predesign.
```text
Author: BENYAMIN RAZAZIYAN
Project: Tall Building Outrigger Predesign Framework
Version: 4.4-reviewed-preliminary-framework
```
---
13. Final Technical Note
The framework is suitable for:
conceptual comparison,
thesis-level parametric studies,
early-stage section estimation,
outrigger effectiveness studies,
preparation before detailed ETABS modeling.
The framework is not suitable as the only basis for final structural design, construction drawings, or code approval.
