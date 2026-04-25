# tall_building_v4_outrigger.app
https://tallbuildingv4outriggerapp-c2gaoqfca7di6woww8pvod.streamlit.app/




Tall Building Outrigger Predesign Framework — Version 4.4
Author: BENYAMIN RAZAZIYAN  
Project type: Preliminary computational framework for tall-building lateral-system comparison  
Main topic: Core–outrigger structural behavior, braced-bay stiffness assignment, modal response-spectrum analysis, and preliminary member sizing  
Status: Research/preliminary design tool; not a replacement for ETABS, SAP2000, SAFE, or a full finite-element design model.
---
1. Purpose of the Code
This Python/Streamlit program was developed to study the preliminary behavior of tall buildings with and without an outrigger system. The main objective is not to perform final code design of every structural member, but to create a defensible computational framework that can estimate how a core–outrigger system changes:
global lateral stiffness,
fundamental period,
modal behavior,
response-spectrum base shear,
story shear distribution,
lateral displacement,
interstory drift,
preliminary wall, column, beam, slab, and outrigger dimensions,
structural quantity trends,
and the relative benefit of adding outriggers at selected stories.
The code is useful for early-stage structural studies, research comparison, thesis work, and preparing a parametric model before detailed ETABS verification.
---
2. Engineering Problem Addressed
Tall buildings are often controlled by lateral stiffness rather than gravity strength alone. In a conventional core-only system, the building behaves similarly to a vertical cantilever. When an outrigger is added, the central core is connected to exterior columns through braced bays, belt/collector elements, and axial force transfer. This creates a coupled lateral-resisting mechanism and can reduce:
lateral drift,
overturning demand on the core,
rotation of the core,
and the fundamental period.
The key issue in many simplified models is that the outrigger is drawn in plan but not correctly included in the global stiffness matrix. This framework corrects that by linking the same braced bays used in the plan drawing to the stiffness calculation and MDOF model.
---
3. General Workflow of the Program
The code follows this sequence:
Read the building geometry, material properties, load assumptions, seismic parameters, and outrigger configuration.
Divide the tower height into lower, middle, and upper zones.
Generate preliminary member dimensions for walls, columns, beams, slabs, and outrigger-related elements.
Enforce rational vertical tapering so lower-story vertical elements are not smaller than upper-story elements.
Calculate story mass, weight, concrete volume, steel quantity, and lateral stiffness.
Build an Euler–Bernoulli flexural MDOF stick model with two DOFs per floor: lateral displacement and rotation.
Add outrigger stiffness at the selected outrigger stories using real selected braced bays.
Solve the generalized eigenvalue problem to obtain periods, frequencies, mode shapes, and modal mass participation.
Perform ASCE-type response-spectrum analysis and equivalent lateral force scaling.
Check drift limits and modal mass participation.
Run an optional redesign loop to improve member sizing.
Produce tables, plots, plan views, response diagrams, diagnostic tables, and an exportable report.
---
4. Main Engineering Assumptions
The model is a preliminary flexural stick model. It is not a full shell/beam/solid finite-element model. The following assumptions are used:
The building is represented as a vertical flexural cantilever-type MDOF system.
Each story node has lateral displacement and rotation DOFs.
The model is solved independently in X and Y directions.
Floor diaphragms are assumed sufficiently rigid for global lateral response.
Core walls, side walls, perimeter walls, and columns contribute to lateral stiffness.
Effective cracked stiffness factors are used for concrete members.
Outrigger stiffness is introduced at selected stories as a coupled stiffness contribution, not only as a visual line in the plan.
Basement retaining walls are included approximately as additional base-zone stiffness when basement levels are defined.
The model is intended for preliminary comparison and parametric study, not final member design.
---
5. Why Modal Analysis Is Used
For tall buildings, the first mode is usually dominant, but higher modes can significantly influence story shear, acceleration, and response-spectrum results. Therefore, the program does not rely only on a single equivalent static shape. It solves the modal eigenvalue problem:
```text
[K] {phi} = omega² [M] {phi}
```
where:
`[K]` is the assembled lateral stiffness matrix,
`[M]` is the mass matrix,
`{phi}` is the mode shape,
`omega` is the circular frequency.
The output includes:
modal periods,
frequencies,
normalized mode shapes,
modal participation factors,
effective modal mass ratios,
cumulative modal mass ratios.
The cumulative modal mass is checked because seismic response-spectrum analysis should include enough modes to represent the dynamic mass of the structure.
---
6. Main Input Groups
6.1 Geometry Inputs
Input	Meaning	Engineering role
`n_story`	Number of above-ground stories	Controls height, mass distribution, stiffness matrix size, and period
`n_basement`	Number of basement stories	Activates approximate retaining-wall stiffness contribution
`story_height`	Typical story height	Controls total height, element length, drift ratio, and modal stiffness
`basement_height`	Basement story height	Used for below-grade stiffness estimation
`plan_x`, `plan_y`	Plan dimensions	Controls floor area, bay length, lever arm, column grid, and outrigger arm
`n_bays_x`, `n_bays_y`	Number of grid bays	Defines column spacing and possible braced-bay locations
`plan_shape`	Basic plan type	Used for floor-area calculation
These inputs define the structural grid and the geometric basis for stiffness, mass, and outrigger load path.
---
6.2 Core and Service Inputs
Input	Meaning	Engineering role
`stair_count`	Number of stairs	Used to estimate minimum required core opening area
`elevator_count`	Number of elevators	Used to estimate service/core demand
`elevator_area_each`	Area per elevator	Controls opening size
`stair_area_each`	Area per stair	Controls opening size
`service_area`	Mechanical/service area	Added to required core area
`corridor_factor`	Circulation allowance	Expands service area for realistic core planning
`core_ratio_x`, `core_ratio_y`	Initial core size ratios	Estimate core dimensions relative to plan
`core_max_ratio_x`, `core_max_ratio_y`	Maximum core size limits	Prevent unrealistically large core dimensions
The code uses these values to estimate a reasonable central core size before calculating wall stiffness.
---
6.3 Material Inputs
Input	Meaning	Engineering role
`fck`	Concrete compressive strength	Affects preliminary member sizing and material-size factor
`Ec`	Concrete elastic modulus	Directly affects lateral stiffness `EI` and therefore periods and drifts
`fy`	Steel yield strength	Used in quantity/design context
`STEEL_E_MPA`	Steel elastic modulus	Used for tubular/braced outrigger stiffness
Increasing `fck` and `Ec` should influence wall, column, beam, slab, and global stiffness trends. However, if a member reaches a minimum practical size limit, its dimension will no longer reduce with higher strength.
---
6.4 Gravity Load and Mass Inputs
Input	Meaning	Engineering role
`DL`	Additional dead load	Added to story seismic weight
`LL`	Live load	Partially included in seismic mass
`live_load_mass_factor`	Live-load mass participation factor	Controls how much live load contributes to mass
`slab_finish_allowance`	Finishes/superimposed dead load	Added to floor weight
`facade_line_load`	Façade line load	Added around building perimeter
`additional_mass_factor`	Global mass multiplier	Used for sensitivity or conservative mass adjustment
These inputs directly affect the mass matrix `[M]`. Since period is proportional to `sqrt(M/K)`, mass assumptions strongly influence the calculated period.
---
6.5 Preliminary Member Size Limits
Input	Meaning	Engineering role
`min_wall_thickness`, `max_wall_thickness`	Wall thickness limits	Prevent unrealistic wall sizing
`min_column_dim`, `max_column_dim`	Column dimension limits	Control practical gravity/lateral member sizing
`min_beam_width`, `min_beam_depth`	Beam size limits	Control preliminary beam dimensions
`min_slab_thickness`, `max_slab_thickness`	Slab thickness limits	Control gravity/diaphragm thickness
These limits are important. If a member reaches its minimum value, increasing material strength may not reduce its size further.
---
6.6 Effective Stiffness Factors
Input	Meaning	Engineering role
`wall_cracked_factor`	Effective stiffness factor for core walls	Reduces gross wall stiffness for cracked concrete behavior
`column_cracked_factor`	Effective stiffness factor for columns	Reduces gross column stiffness
`side_wall_cracked_factor`	Effective stiffness factor for side/perimeter walls	Used in wall contribution
`coupling_factor`	Global coupling adjustment	Adjusts combined stiffness participation
These factors are necessary because reinforced-concrete members do not behave with full gross elastic stiffness under seismic lateral loading.
---
6.7 Wall and Column Layout Inputs
Input	Meaning	Engineering role
`lower_zone_wall_count`	Wall count in lower zone	Helps define preliminary wall distribution
`middle_zone_wall_count`	Wall count in middle zone	Helps define vertical stiffness transition
`upper_zone_wall_count`	Wall count in upper zone	Helps define upper-tower wall layout
`perimeter_column_factor`	Perimeter column size multiplier	Makes perimeter columns stronger than interior columns
`corner_column_factor`	Corner column size multiplier	Makes corner columns stronger than other columns
`side_wall_ratio`	Side-wall length ratio	Defines side-wall lengths relative to plan
`perimeter_wall_ratio`	Perimeter-wall participation ratio	Defines additional perimeter wall contribution
These inputs are used to avoid a pure core-only model and to represent the contribution of columns, side walls, and perimeter walls.
---
6.8 Outrigger Inputs
Input	Meaning	Engineering role
`outrigger_system`	None, tubular bracing, or belt truss	Defines whether outrigger action exists
`outrigger_count`	Number of outrigger levels	Controls how many selected stories are active
`outrigger_story_levels`	Candidate outrigger stories	Defines the vertical location of outriggers
`braced_spans_x`, `braced_spans_y`	Number of braced bays in each direction	Controls how many real bay panels are braced
`braced_bay_ids_x`, `braced_bay_ids_y`	Optional exact bay IDs	Allows user-defined braced-bay locations instead of automatic central selection
`outrigger_depth_m`	Vertical/depth parameter of outrigger brace geometry	Affects brace length and angle
`tubular_diameter_m`	Tube diameter	Affects brace steel area
`tubular_thickness_m`	Tube wall thickness	Affects brace steel area
`outrigger_connection_efficiency`	Connection/load-transfer efficiency	Reduces ideal stiffness for realistic connection behavior
The code selects real braced bays and uses the same selected bays in:
plan drawing,
stiffness calculation,
global MDOF matrix assembly,
outrigger diagnostic tables.
---
6.9 Seismic Spectrum Inputs
The program supports two seismic input approaches.
Direct Design Spectrum Input
Input	Meaning
`SDS`	Design spectral acceleration at short period
`SD1`	Design spectral acceleration at 1 second
Site-Coefficient-Based Input
Input	Meaning
`SS`	Mapped spectral acceleration at short period
`S1`	Mapped spectral acceleration at 1 second
`Fa`	Short-period site coefficient
`Fv`	Long-period site coefficient
`use_site_coefficients`	If true, `SDS` and `SD1` are calculated from `SS`, `S1`, `Fa`, and `Fv`
When site coefficients are used:
```text
SDS = (2/3) Fa SS
SD1 = (2/3) Fv S1
```
Other seismic parameters:
Input	Meaning	Engineering role
`R`	Response modification factor	Reduces elastic force demand
`Ie`	Importance factor	Adjusts seismic demand
`Cd`	Deflection amplification factor	Used for design drift amplification
`TL`	Long-period transition period	Defines spectrum long-period branch
`damping_ratio`	Modal damping ratio	Used in CQC modal combination
`Ct`, `x_exp`	Approximate period coefficients	Used for approximate code period `Ta`
`Cu`	Upper-bound period coefficient	Used for ELF period cap if selected
`rsa_min_ratio_to_elf`	Minimum RSA-to-ELF scaling ratio	Scales response-spectrum base shear
---
7. How the Outrigger Stiffness Is Calculated
The outrigger is not treated only as a drawing object. Each selected braced bay contributes axial stiffness. For a single diagonal brace, the basic axial stiffness is:
```text
k_axial = EA / L
```
Because only the horizontal component contributes to lateral coupling, the effective contribution is:
```text
k_span = (EA / L) cos²(theta)
```
The total outrigger stiffness is obtained by summing the stiffness of all active braces:
```text
K_out = Σ (EA / L) cos²(theta)
```
The program then converts this span stiffness into a coupled floor stiffness contribution using the lever arm between the core and exterior braced bays. The key concept is that the outrigger resists both lateral displacement and core rotation. Therefore, the stiffness contribution is assembled as a coupled matrix rather than a simple isolated spring.
Conceptual form:
```text
K_add = K_out [ 1      a
                a     a² ]
```
where `a` is the effective lever arm. This is why the outrigger can influence global stiffness and period.
---
8. What the Code Does Section by Section
8.1 Data Models
The code defines structured data classes for:
seismic parameters,
building input,
story sections,
story properties,
modal results,
response-spectrum results,
and final design result.
This makes the model traceable and easier to audit.
---
8.2 Geometry and Section Sizing
The geometry module:
divides the building into lower, middle, and upper zones,
calculates service/core opening requirements,
estimates central core dimensions,
sizes walls, columns, beams, and slabs,
applies material-strength factors,
and enforces monotonic vertical member sizing.
The monotonic rule prevents illogical outcomes such as a middle-story column becoming larger than a lower-story column without a structural reason.
---
8.3 Section Property Calculation
The section-property module calculates:
core-wall inertia,
side-wall inertia,
perimeter-wall inertia,
basement retaining-wall influence,
column group inertia,
outrigger brace area,
outrigger span stiffness,
story mass and weight,
concrete quantity,
steel quantity,
effective lateral stiffness in X and Y directions.
For X-direction lateral motion, bending is associated with the corresponding orthogonal inertia component. The same logic is applied independently for Y-direction motion.
---
8.4 MDOF Flexural Solver
The solver uses Euler–Bernoulli beam-column elements along the height of the building. Each floor has:
lateral displacement DOF,
rotational DOF.
The base is fixed. The global mass and stiffness matrices are assembled and then solved using a generalized eigenvalue procedure.
---
8.5 Response-Spectrum Analysis
The response-spectrum module:
calculates spectral acceleration,
computes modal floor forces,
combines modal responses using CQC or SRSS,
calculates story shear,
calculates overturning moment,
calculates displacement,
calculates interstory drift,
scales RSA base shear to a minimum ELF ratio,
applies drift amplification using `Cd/Ie`.
---
8.6 Redesign Loop
The optional redesign loop checks:
maximum drift ratio,
minimum modal mass participation,
member size limits,
material quantity trends.
If the drift exceeds the limit, the program increases wall, column, beam, slab, and outrigger scale factors. If the model is excessively stiff, it can reduce selected scale factors within safe bounds.
---
9. Output Tables
The code produces several engineering tables.
9.1 Summary Table
Includes:
project title,
author name,
version,
building height,
floor area,
seismic weight,
total mass,
fundamental period in X and Y,
modal mass participation,
base shear,
maximum drift ratio,
concrete quantity,
steel quantity,
selected outrigger system.
---
9.2 Final Dimensions Table
Reports representative member dimensions for lower, middle, and upper zones:
core dimensions,
core opening dimensions,
core wall thickness,
side-wall dimensions,
interior column size,
perimeter column size,
corner column size,
beam size,
slab thickness.
---
9.3 Modal Table
Reports for each mode:
mode number,
direction,
period,
frequency,
modal participation factor,
effective modal mass percentage,
cumulative modal mass percentage.
This table is important for checking whether enough modes are included.
---
9.4 Story Response Table
Reports story-by-story seismic response:
floor force,
story shear,
overturning moment,
displacement,
interstory drift,
drift ratio.
This is the main table for comparing drift and force distribution.
---
9.5 Stiffness Table
Reports:
effective story stiffness indicators,
outrigger rotational stiffness,
outrigger lateral stiffness,
mass,
concrete quantity,
steel quantity.
This table is used to verify whether outriggers, walls, and columns actually influence the stiffness model.
---
9.6 Outrigger Diagnostic Table
This is one of the most important outputs. It reports:
active outrigger levels,
active braced bay IDs,
number of selected braced spans,
number of braces,
brace length,
brace angle factor,
one-brace stiffness,
total span stiffness,
final stiffness contribution inserted into the model.
This table should be checked whenever the outrigger effect appears too small or too large.
---
9.7 Outrigger Effect Comparison
The code compares the same building with and without outrigger action. The comparison includes:
period without outrigger,
period with outrigger,
percentage period reduction,
drift without outrigger,
drift with outrigger,
drift reduction,
base shear comparison.
This is the most useful output for thesis or paper discussion.
---
10. Output Figures
The program can generate:
structural plan view,
braced-bay outrigger layout,
perimeter wall layout,
core and column grid,
modal shape plots,
story displacement/drift/shear plots,
stiffness distribution plots,
redesign iteration plots,
response spectrum plot.
The plan view is intended to show the actual structural logic used in calculation: selected braced bays should match the stiffness diagnostic table.
---
11. How to Run the Program
11.1 Install Requirements
```bash
pip install numpy pandas matplotlib scipy streamlit
```
11.2 Run with Streamlit
```bash
streamlit run tower_outrigger_predesign_framework_v4_4_reviewed.py
```
11.3 Typical Use Procedure
Enter building geometry.
Enter material properties.
Define gravity load and mass assumptions.
Define seismic spectrum parameters.
Select outrigger system type.
Define outrigger count and outrigger story levels.
Define braced spans or exact braced bay IDs.
Run the model.
Check the summary table.
Check modal mass participation.
Check outrigger diagnostic table.
Compare with and without outrigger.
Export or copy tables for ETABS comparison.
---
12. Recommended ETABS Verification
For article or thesis use, the following ETABS checks are recommended:
Framework output	ETABS comparison
Fundamental period X/Y	Modal analysis periods
Mode shapes	Modal deformed shapes
Modal mass participation	ETABS modal participating mass ratios
Base shear	Response-spectrum base reactions
Story shear	Story force output
Drift ratio	Story drift output
Outrigger effect	Model with/without outrigger cases
Plan braced bays	ETABS frame elevation and plan bracing layout
The framework results should be interpreted as preliminary trends. Exact matching with ETABS is not expected because ETABS uses a full 3D member model, while this code uses a reduced-order flexural stick representation.
---
13. Engineering Limitations
This code has important limitations that must be stated in any article or thesis.
It is not a final design program.
It does not design reinforcement according to ACI, Eurocode, or Turkish code member-detailing provisions.
It does not model full 3D torsional irregularity.
It assumes simplified diaphragm behavior.
It does not explicitly model soil–structure interaction.
Basement wall stiffness is approximate.
Outrigger stiffness is based on equivalent brace-span stiffness, not a detailed finite-element connection model.
Beam-column joint flexibility is not explicitly modeled.
P–Delta effects are not fully implemented as a nonlinear geometric stiffness formulation.
Construction sequence, creep, shrinkage, and long-term effects are not included.
Final drift and strength checks must be verified in ETABS or another professional structural analysis program.
---
14. Correct Interpretation for Research Use
The correct academic description is:
> This tool is a preliminary computational framework for evaluating the influence of outrigger and braced-bay configurations on the global dynamic response of tall buildings. It estimates relative changes in stiffness, period, modal response, drift, and preliminary member dimensions. The results are intended for early-stage design comparison and must be validated using a full three-dimensional structural analysis model.
Recommended article wording:
> The proposed framework is not intended to replace detailed finite-element analysis. Instead, it provides a transparent parametric environment for preliminary assessment of core–outrigger systems and for identifying efficient outrigger configurations before detailed ETABS verification.
---
15. What the Inputs Help Us Understand
The input parameters allow the user to study how different design decisions affect the global behavior of a tall building.
Increasing `n_story` usually increases period and drift demand.
Increasing `Ec` increases stiffness and can reduce drift and period.
Increasing `fck` can reduce required preliminary member dimensions unless minimum limits control.
Increasing wall thickness increases stiffness and reduces drift.
Increasing column sizes increases frame contribution and lateral stiffness.
Increasing outrigger count generally increases lateral stiffness if placed effectively.
Moving outrigger levels changes modal behavior and drift profile.
Increasing braced spans increases outrigger stiffness.
Increasing brace diameter or thickness increases `EA/L` stiffness.
Increasing mass increases period and seismic inertia demand.
Changing `SDS`, `SD1`, `SS`, `S1`, `Fa`, and `Fv` changes spectral demand.
Changing `R`, `Ie`, and `Cd` changes force reduction and drift amplification.
---
16. What the Outputs Tell Us
The outputs answer the following engineering questions:
Is the building too flexible?
Does the outrigger reduce the fundamental period?
Does the outrigger reduce maximum drift?
Are enough modes included in the analysis?
Which direction is more critical, X or Y?
How much base shear is generated by the response spectrum?
Which stories have high drift concentration?
Are selected outrigger bays actually active in the stiffness model?
Do member sizes change rationally with height?
Does material strength influence member dimensions and stiffness?
How different is the building response with and without outriggers?
---
