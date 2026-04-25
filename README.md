# Tall Building Core–Outrigger Predesign Framework v4.4
## 🔗 Live Application

👉 https://tallbuildingv4outriggerapp-c2gaoqfca7di6woww8pvod.streamlit.app/
**Author:** BENYAMIN RAZAZIYAN  
**Version:** 4.4  
**Project type:** Preliminary computational framework for tall-building lateral-system comparison  
**Main scope:** Core wall, perimeter/side walls, basement retaining-wall contribution, braced-bay outrigger stiffness, modal analysis, response-spectrum analysis, drift check, and preliminary member sizing.

\---

## 1\. Purpose of the Program

This program is a Python/Streamlit-based preliminary structural analysis framework for tall buildings with and without outrigger systems. The purpose is to evaluate how the structural system changes when a core wall system is coupled to exterior columns through braced outrigger bays.

The program estimates:

* global lateral stiffness,
* fundamental periods in X and Y directions,
* mode shapes and modal mass participation,
* response-spectrum base shear,
* story shear and overturning response,
* lateral displacement and interstory drift,
* preliminary dimensions of walls, columns, beams, slabs, and outriggers,
* the stiffness contribution of selected outrigger braced bays,
* and comparative response with and without outriggers.

The program is intended for **preliminary comparison, parametric study, and research-level computational framework development**. It is not a replacement for ETABS, SAP2000, SAFE, or full three-dimensional finite-element design.

\---

## 2\. Engineering Idea Behind the Model

A tall building without an outrigger usually behaves like a vertical cantilever dominated by the flexural stiffness of the central core. When an outrigger is added, the core rotation is restrained by axial forces developed in exterior columns through outrigger braces and belt/collector action.

The intended load path is:

**Core rotation → outrigger arms → selected braced bays → exterior columns/perimeter system → global lateral stiffness increase**

Therefore, the outrigger must not be treated only as a drawn plan element. Its stiffness must enter the global stiffness matrix of the MDOF model. This code assigns stiffness to the same braced bays that are shown in the plan view, so the graphical model and numerical model remain consistent.

\---

## 3\. Overall Computational Workflow

The program follows this sequence:

1. Read geometry, story, material, load, seismic, and outrigger inputs.
2. Generate the structural grid from plan dimensions and number of bays.
3. Estimate core dimensions from service/core requirements.
4. Size preliminary walls, columns, beams, and slabs.
5. Enforce rational vertical member tapering, so lower-story members are not smaller than upper-story members.
6. Compute mass, concrete quantity, steel quantity, and lateral stiffness for each story.
7. Select active outrigger stories based on `outrigger\_count` and `outrigger\_story\_levels`.
8. Select actual braced bays between grid columns.
9. Compute outrigger stiffness from brace axial stiffness.
10. Add the outrigger stiffness as a coupled stiffness matrix to the global MDOF system.
11. Assemble the global mass and stiffness matrices.
12. Solve the modal eigenvalue problem.
13. Perform ASCE-style response-spectrum analysis.
14. Apply ELF/RSA base-shear scaling if required.
15. Compute displacement, drift, story shear, overturning, and diagnostic tables.
16. Produce plan/elevation plots and comparison outputs.

\---

## 4\. Structural Idealization Used in the Code

The building is idealized as a vertical flexural stick model with two degrees of freedom at each floor level:

* lateral displacement, (u),
* floor rotation, (\\theta).

The model is solved independently in X and Y directions.

For a building with (n) stories, the full model initially has (2(n+1)) degrees of freedom, including the fixed base node. After fixing base displacement and base rotation, the free system has (2n) active degrees of freedom.

The floor DOF vector has the form:

$$
\\mathbf{q}=\\begin{bmatrix}
u\_1 \& \\theta\_1 \& u\_2 \& \\theta\_2 \& \\cdots \& u\_n \& \\theta\_n
\\end{bmatrix}^T
$$

where (u\_i) is the lateral displacement and (\\theta\_i) is the rotation at story level (i).

\---

## 5\. Story Flexural Element Stiffness

Each story segment is modeled as an Euler–Bernoulli beam-column flexural element. For a story height (L) and effective flexural rigidity (EI), the local element stiffness matrix is:

$$
\\mathbf{k}\_e=\\frac{EI}{L^3}
\\begin{bmatrix}
12 \& 6L \& -12 \& 6L \\
6L \& 4L^2 \& -6L \& 2L^2 \\
-12 \& -6L \& 12 \& -6L \\
6L \& 2L^2 \& -6L \& 4L^2
\\end{bmatrix}
$$

This matrix is assembled story by story into the global stiffness matrix (\\mathbf{K}).

For X-direction translation, the program uses the effective flexural stiffness about the Y-axis. For Y-direction translation, it uses the effective flexural stiffness about the X-axis.

\---

## 6\. Effective Lateral Stiffness Components

The total story stiffness is assembled from several stiffness contributors:

|Component|Role in the model|
|-|-|
|Core wall|Main flexural lateral-resisting component|
|Side/perimeter walls|Additional distributed wall stiffness and plan restraint|
|Columns|Gravity and lateral stiffness contribution through the column grid|
|Beams/slabs|Mass, quantity, collector action, and stiffness-related sizing influence|
|Basement retaining walls|Approximate below-grade stiffness contribution when basements exist|
|Outrigger braces|Coupled stiffness transfer between core and exterior columns|

The cracked stiffness factors are used to reduce gross section stiffness for preliminary reinforced-concrete behavior:

|Parameter|Meaning|
|-|-|
|`wall\_cracked\_factor`|Effective stiffness factor for core walls|
|`column\_cracked\_factor`|Effective stiffness factor for columns|
|`side\_wall\_cracked\_factor`|Effective stiffness factor for side/perimeter walls|
|`coupling\_factor`|Global coupling adjustment factor for lateral stiffness|

These values are preliminary modeling factors and should be calibrated against ETABS or project-specific assumptions.

\---

## 7\. Why Modal Analysis Is Used

For tall buildings, the first mode is often dominant, but higher modes may significantly affect story shear, acceleration, and response-spectrum demand. Therefore, the program does not rely only on an equivalent static triangular force pattern.

The modal eigenvalue problem solved by the program is:

$$
\\mathbf{K}\\boldsymbol{\\phi}\_r=\\omega\_r^2\\mathbf{M}\\boldsymbol{\\phi}\_r
$$

where:

|Symbol|Meaning|
|-|-|
|(\\mathbf{K})|assembled global stiffness matrix|
|(\\mathbf{M})|assembled mass matrix|
|(\\boldsymbol{\\phi}\_r)|mode shape of mode (r)|
|(\\omega\_r)|circular natural frequency of mode (r)|

The natural period of each mode is:

$$
T\_r=\\frac{2\\pi}{\\omega\_r}
$$

The program reports:

* modal period,
* modal frequency,
* normalized mode shape,
* modal participation factor,
* effective modal mass ratio,
* cumulative modal mass ratio.

Cumulative modal mass is checked because response-spectrum analysis should include enough modes to represent the dynamic mass of the structure.

\---

## 8\. Modal Participation and Effective Mass

For each mode, the modal participation factor is calculated as:

$$
\\Gamma\_r=\\frac{\\boldsymbol{\\phi}\_r^T\\mathbf{M}\\mathbf{r}}
{\\boldsymbol{\\phi}\_r^T\\mathbf{M}\\boldsymbol{\\phi}\_r}
$$

where (\\mathbf{r}) is the influence vector for lateral translation.

The effective modal mass is:

$$
M\_{eff,r}=\\Gamma\_r^2\\left(\\boldsymbol{\\phi}\_r^T\\mathbf{M}\\boldsymbol{\\phi}\_r\\right)
$$

The effective modal mass ratio is:

$$
\\eta\_r=\\frac{M\_{eff,r}}{M\_{total}}
$$

and the cumulative modal mass ratio is:

$$
\\eta\_{cum}=\\sum\_{r=1}^{m}\\eta\_r
$$

The default target is controlled by `minimum\_modal\_mass\_ratio`, commonly set to 0.90 for preliminary dynamic adequacy.

\---

## 9\. Outrigger Stiffness Formulation

The outrigger is assigned to actual braced bays between columns, not randomly spread across the plan. Each X-braced bay contains two diagonal braces. The stiffness of one diagonal brace projected into the relevant horizontal direction is calculated as:

$$
k\_{brace}=\\frac{E\_sA\_b}{L\_b}\\cos^2\\theta
$$

where:

|Symbol|Meaning|
|-|-|
|(E\_s)|modulus of elasticity of steel|
|(A\_b)|cross-sectional area of one brace|
|(L\_b)|diagonal brace length|
|(\\theta)|brace angle relative to the horizontal direction|

For one braced bay:

$$
L\_b=\\sqrt{b^2+h\_o^2}
$$

$$
\\cos^2\\theta=\\left(\\frac{b}{L\_b}\\right)^2
$$

where (b) is the bay width and (h\_o) is the outrigger depth.

The total bracing stiffness is obtained by multiplying the stiffness of one brace by the number of braces:

$$
K\_{out}=N\_{brace},k\_{brace}
$$

For X-bracing on two opposite sides of the building:

$$
N\_{brace}=2\\times2\\times N\_{bay}
$$

where:

* the first 2 represents two diagonals per X-braced bay,
* the second 2 represents two opposite sides of the plan,
* (N\_{bay}) is the number of selected braced bays.

\---

## 10\. Coupled Outrigger Matrix Added to the MDOF System

The important correction in this version is that the outrigger is not added only as a simple rotational spring. The outrigger stiffness is inserted into the floor stiffness matrix as a coupled matrix acting on displacement and rotation.

The compatibility relation is assumed as:

$$
\\delta=u+a\\theta
$$

where:

|Symbol|Meaning|
|-|-|
|(u)|lateral displacement at the outrigger floor|
|(\\theta)|rotation at the outrigger floor|
|(a)|lever arm from core to exterior outrigger line|

The strain energy of the outrigger spring is:

$$
U=\\frac{1}{2}\\eta K\_{out}(u+a\\theta)^2
$$

Therefore, the condensed outrigger stiffness matrix added to the floor DOFs (\[u,\\theta]^T) is:

$$
\\mathbf{K}*{out,floor}=\\eta K*{out}
\\begin{bmatrix}
1 \& a \\
a \& a^2
\\end{bmatrix}
$$

This matrix contributes:

* direct lateral stiffness, (K\_{uu}),
* displacement–rotation coupling stiffness, (K\_{u\\theta}),
* rotational restraint, (K\_{\\theta\\theta}).

This is why the outrigger can affect the global stiffness matrix, fundamental period, displacement, and drift.

\---

## 11\. Active Outrigger Levels and Braced Bays

The program uses two levels of outrigger definition:

|Input|Meaning|
|-|-|
|`outrigger\_count`|Number of active outrigger floors|
|`outrigger\_story\_levels`|Candidate story levels where outriggers may be placed|
|`braced\_spans\_x`|Number of selected braced bays for X-direction outrigger action|
|`braced\_spans\_y`|Number of selected braced bays for Y-direction outrigger action|
|`braced\_bay\_ids\_x`|Optional explicit bay IDs for X-direction braced bays|
|`braced\_bay\_ids\_y`|Optional explicit bay IDs for Y-direction braced bays|

If explicit bay IDs are not provided, the code selects central/symmetric braced bays automatically. The same selected bay IDs are used in:

* outrigger stiffness calculation,
* plan drawing,
* diagnostic output tables,
* global MDOF matrix assembly.

This prevents the previous problem where braces were visually shown but did not correspond to the stiffness model.

\---

## 12\. Response-Spectrum Calculation

The code supports two seismic spectrum input modes:

### 12.1 Direct Design-Spectrum Input

The user directly provides:

* `SDS`,
* `SD1`.

These values are used directly in the design spectrum.

### 12.2 Site-Coefficient-Based Input

The user provides:

* `SS`,
* `S1`,
* `Fa`,
* `Fv`.

The program then calculates:

$$
S\_{DS}=\\frac{2}{3}F\_aS\_S
$$

$$
S\_{D1}=\\frac{2}{3}F\_vS\_1
$$

The corner periods are:

$$
T\_s=\\frac{S\_{D1}}{S\_{DS}}
$$

$$
T\_0=0.2T\_s
$$

The elastic design spectral acceleration shape is calculated as:

$$
S\_a(T)=
\\begin{cases}
S\_{DS}\\left(0.4+0.6\\frac{T}{T\_0}\\right), \& T<T\_0 \\
S\_{DS}, \& T\_0\\leq T\\leq T\_s \\
\\frac{S\_{D1}}{T}, \& T\_s<T\\leq T\_L \\
\\frac{S\_{D1}T\_L}{T^2}, \& T>T\_L
\\end{cases}
$$

For force calculation, the spectrum is reduced using:

$$
S\_{a,design}(T)=S\_a(T)\\frac{I\_e}{R}
$$

where (R) is the response modification factor and (I\_e) is the importance factor.

\---

## 13\. Modal Response Calculation

For each mode, the approximate modal displacement response is calculated using:

$$
\\mathbf{u}\_r=\\boldsymbol{\\phi}\_r\\Gamma\_r\\frac{S\_a}{\\omega\_r^2}
$$

The modal lateral force vector is:

$$
\\mathbf{F}\_r=\\mathbf{M}\\boldsymbol{\\phi}\_r\\Gamma\_rS\_a
$$

Story shear is calculated by accumulating floor forces from top to bottom:

$$
V\_i=\\sum\_{j=i}^{n}F\_j
$$

Interstory drift is obtained from floor displacement differences:

$$
\\Delta\_i=u\_i-u\_{i-1}
$$

The design drift is amplified by:

$$
\\Delta\_{design}=\\Delta\\frac{C\_d}{I\_e}
$$

\---

## 14\. Modal Combination

The program supports two modal combination methods:

|Method|Use|
|-|-|
|SRSS|Suitable when modal frequencies are well separated|
|CQC|More suitable for closely spaced modes and tall-building behavior|

The CQC correlation coefficient used in the code is:

$$
\\rho\_{ij}=\\frac{8\\zeta^2\\beta^{3/2}}
{(1-\\beta^2)^2+4\\zeta^2\\beta(1+\\beta)^2}
$$

where:

$$
\\beta=\\frac{\\omega\_j}{\\omega\_i}
$$

and (\\zeta) is the damping ratio.

\---

## 15\. ELF Base-Shear Check and Scaling

The program calculates an equivalent lateral force base shear for comparison and possible scaling of response-spectrum results.

The approximate period is:

$$
T\_a=C\_t h\_n^x
$$

If the period cap is active:

$$
T\_{used}=\\min(T\_{modal},C\_uT\_a)
$$

The seismic response coefficient is calculated using:

$$
C\_s=\\min\\left(\\frac{S\_{DS}}{R/I\_e},\\frac{S\_{D1}}{T(R/I\_e)}\\right)
$$

with a lower bound:

$$
C\_s\\geq \\max(0.044S\_{DS}I\_e,0.01)
$$

The ELF base shear is:

$$
V=C\_sW
$$

The response-spectrum base shear may be scaled to a target ratio of ELF base shear using `rsa\_min\_ratio\_to\_elf`.

\---

## 16\. Main Input Groups

### 16.1 Geometry Inputs

|Input|Meaning|Engineering effect|
|-|-|-|
|`n\_story`|Number of above-ground stories|Controls height, mass, matrix size, and period|
|`n\_basement`|Number of basement stories|Activates approximate retaining-wall stiffness contribution|
|`story\_height`|Typical story height|Controls element length, drift ratio, and modal stiffness|
|`basement\_height`|Basement story height|Used in below-grade stiffness approximation|
|`plan\_x`, `plan\_y`|Plan dimensions|Control floor area, lever arm, bay length, and mass|
|`n\_bays\_x`, `n\_bays\_y`|Grid bay counts|Define column grid and possible braced-bay locations|
|`plan\_shape`|Basic plan type|Used for approximate floor-area calculation|

### 16.2 Core and Service Inputs

|Input|Meaning|Engineering effect|
|-|-|-|
|`stair\_count`|Number of stair cores|Affects minimum core service area|
|`elevator\_count`|Number of elevators|Affects minimum core service area|
|`elevator\_area\_each`|Area assigned to each elevator|Affects opening/core demand|
|`stair\_area\_each`|Area assigned to each stair|Affects opening/core demand|
|`service\_area`|Additional service area|Affects core opening estimate|
|`corridor\_factor`|Circulation multiplier|Enlarges service/core requirement|
|`core\_ratio\_x`, `core\_ratio\_y`|Core size ratios|Control core dimensions|
|`core\_max\_ratio\_x`, `core\_max\_ratio\_y`|Maximum core ratio|Prevents unrealistically large core dimensions|

### 16.3 Material Inputs

|Input|Meaning|Engineering effect|
|-|-|-|
|`fck`|Concrete compressive strength|Affects preliminary member sizing and strength-controlled dimensions|
|`Ec`|Concrete elastic modulus|Directly affects global stiffness and period|
|`fy`|Steel yield strength|Used for reinforcement/steel-related assumptions|
|`reference\_fck`|Reference strength|Used for material-responsive sizing comparison|
|`reference\_Ec`|Reference modulus|Used for stiffness-responsive preliminary sizing|

Increasing `fck` and `Ec` should influence wall, column, beam, and slab sizing unless the section reaches minimum practical limits.

### 16.4 Load and Mass Inputs

|Input|Meaning|Engineering effect|
|-|-|-|
|`DL`|Superimposed dead load|Increases seismic mass|
|`LL`|Live load|Partially contributes to seismic mass|
|`live\_load\_mass\_factor`|Live load participation factor|Controls how much live load enters mass|
|`slab\_finish\_allowance`|Floor finish load|Adds mass|
|`facade\_line\_load`|Perimeter façade load|Adds mass at each floor|
|`additional\_mass\_factor`|Global mass multiplier|Used for sensitivity/calibration|

### 16.5 Preliminary Section Limits

|Input|Meaning|
|-|-|
|`min\_wall\_thickness`, `max\_wall\_thickness`|Practical bounds for wall thickness|
|`min\_column\_dim`, `max\_column\_dim`|Practical bounds for column dimensions|
|`min\_beam\_width`, `min\_beam\_depth`|Practical minimum beam size|
|`min\_slab\_thickness`, `max\_slab\_thickness`|Practical slab thickness bounds|

These limits explain why increasing material strength may not always reduce dimensions: once a member reaches a minimum practical bound, it cannot reduce further.

### 16.6 Outrigger Inputs

|Input|Meaning|Engineering effect|
|-|-|-|
|`outrigger\_system`|None, tubular bracing, or belt truss|Selects outrigger stiffness model|
|`outrigger\_count`|Number of active outrigger levels|Controls how many stories receive outrigger stiffness|
|`outrigger\_story\_levels`|Candidate outrigger story numbers|Defines where outrigger systems may exist|
|`outrigger\_depth\_m`|Vertical/depth dimension of brace panel|Controls brace length and angle|
|`tubular\_diameter\_m`|Tube diameter|Controls brace area and stiffness|
|`tubular\_thickness\_m`|Tube wall thickness|Controls brace area and stiffness|
|`braced\_spans\_x`, `braced\_spans\_y`|Number of braced bays|Controls total brace count and stiffness|
|`outrigger\_connection\_efficiency`|Connection/load-path efficiency|Reduces ideal brace stiffness for preliminary realism|

\---

## 17\. Main Outputs

### 17.1 Summary Table

The summary table gives:

* total height,
* floor area,
* total seismic weight,
* total mass,
* fundamental period in X and Y,
* modal mass participation,
* base shear in X and Y,
* maximum drift ratio,
* total concrete quantity,
* total steel quantity,
* selected outrigger system.

### 17.2 Final Dimensions Table

This table reports lower, middle, and upper zone dimensions:

* core dimensions,
* core wall thickness,
* side/perimeter wall dimensions,
* interior column size,
* perimeter column size,
* corner column size,
* beam size,
* slab thickness.

### 17.3 Modal Tables

The modal tables report for each mode:

* mode number,
* period,
* frequency,
* participation factor,
* effective modal mass,
* cumulative modal mass.

### 17.4 Story Response Tables

For each story, the program reports:

* elevation,
* floor force,
* story shear,
* overturning moment,
* displacement,
* interstory drift,
* drift ratio.

### 17.5 Outrigger Diagnostic Tables

The outrigger tables are especially important for checking whether the outrigger is numerically active. They report:

* active outrigger stories,
* selected braced bay IDs,
* number of selected bays,
* number of braces,
* brace length,
* one-brace stiffness,
* total brace stiffness,
* lever arm,
* rotational stiffness contribution,
* coupled matrix contribution.

### 17.6 Plots

The code can produce:

* plan layout,
* braced-bay/outtrigger layout,
* mode shapes,
* story displacement,
* drift ratio distribution,
* story shear distribution,
* stiffness/mass diagnostic plots.

\---

## 18\. What Should Be Compared with ETABS

For research validation, ETABS should be used to check:

|Quantity|Expected comparison|
|-|-|
|Fundamental period|Same order of magnitude; exact match is not expected|
|Mode shape|Similar first-mode deformation pattern|
|Modal mass ratio|Similar participation trend|
|Base shear|Similar after spectrum and scaling assumptions are aligned|
|Drift profile|Similar shape and relative reduction due to outrigger|
|Outrigger effect|ETABS should confirm reduction in drift/period when outriggers are added|
|Story shear|Similar trend; local differences are expected|

The comparison must use consistent:

* story mass,
* cracked stiffness modifiers,
* boundary conditions,
* spectrum parameters,
* response modification factor,
* damping ratio,
* diaphragm assumptions,
* outrigger member properties,
* and braced-bay locations.

\---

## 19\. Important Limitations

This program is a preliminary framework. The following limitations must be stated in any thesis, paper, or GitHub repository:

1. The model is a simplified 2D flexural stick model, not a full 3D finite-element building model.
2. Torsional irregularity is not fully captured.
3. Floor diaphragm flexibility is not explicitly modeled.
4. Beam-column joint behavior is simplified.
5. Soil–structure interaction is not included.
6. Basement retaining-wall stiffness is approximate.
7. Outrigger connection efficiency is a preliminary calibration factor.
8. Reinforcement design is not performed in full code-check format.
9. P–Delta effects are not fully modeled.
10. Nonlinear behavior, yielding, cracking progression, and performance-based hinge modeling are outside the present scope.
11. Final member design must be checked in ETABS/SAP2000 and according to the governing design code.

\---

## 20\. Suitable Use in a Paper or Thesis

A suitable description is:

> A Python-based preliminary computational framework was developed to evaluate the dynamic and stiffness effects of braced-bay outrigger systems in tall buildings. The building was idealized as a flexural multi-degree-of-freedom model with two degrees of freedom per floor. The outrigger stiffness was calculated from the axial stiffness of selected X-braced bays and introduced into the global stiffness matrix through a condensed displacement–rotation coupling matrix. The framework was used for parametric comparison of structural periods, modal mass participation, base shear, displacement, and interstory drift. The results are intended for preliminary comparison and must be verified using a detailed three-dimensional ETABS model.

\---

## 21\. Installation

Recommended Python version:

```bash
python 3.10+
```

Install required packages:

```bash
pip install numpy pandas matplotlib scipy streamlit
```

Run the Streamlit app:

```bash
streamlit run tower\_outrigger\_predesign\_framework\_v4\_4\_reviewed.py
```

\---

## 

