import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, sqrt
import streamlit as st
from scipy.linalg import eigh

# ==============================
# DATA MODELS
# ==============================

class Material:
    def __init__(self, E, fy):
        self.E = E
        self.fy = fy


class CoreWall:
    def __init__(self, t, lx, ly):
        self.t = t
        self.lx = lx
        self.ly = ly

    def inertia_x(self):
        return self.t * self.lx**3 / 12

    def inertia_y(self):
        return self.t * self.ly**3 / 12


class Column:
    def __init__(self, b, h):
        self.b = b
        self.h = h

    def inertia(self):
        return self.b * self.h**3 / 12


class Tower:
    def __init__(self, n, h, mass, core, col, E):
        self.n = n
        self.h = h
        self.mass = mass
        self.core = core
        self.col = col
        self.E = E


# ==============================
# STIFFNESS MODEL (IMPROVED)
# ==============================

def story_stiffness(tower):
    k = []
    for i in range(tower.n):
        I_core = tower.core.inertia_x()
        I_col = tower.col.inertia()
        k_story = 12 * tower.E * (I_core + I_col) / tower.h**3
        k.append(k_story)
    return np.array(k)


def assemble_K(k):
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] = k[i]
        else:
            K[i, i] += k[i]
            K[i, i-1] -= k[i]
            K[i-1, i] -= k[i]
            K[i-1, i-1] += k[i]
    return K


def assemble_M(n, m):
    return np.eye(n) * m


# ==============================
# ANALYSIS
# ==============================

def analyze(tower):
    k = story_stiffness(tower)
    K = assemble_K(k)
    M = assemble_M(tower.n, tower.mass)

    w2, phi = eigh(K, M)
    w = np.sqrt(w2)
    T = 2 * pi / w

    return T, phi


# ==============================
# DRIFT
# ==============================

def drift(tower):
    k = story_stiffness(tower)
    K = assemble_K(k)

    f = np.linspace(1, tower.n, tower.n)
    u = np.linalg.solve(K, f)

    drift = np.diff(np.insert(u, 0, 0)) / tower.h
    return u, drift


# ==============================
# STREAMLIT UI
# ==============================

st.title("Tall Building + Outrigger (Simplified Framework)")

n = st.slider("Number of stories", 5, 80, 40)
h = st.slider("Story height (m)", 2.5, 5.0, 3.2)
mass = st.number_input("Floor mass (kg)", value=800000)

core_t = st.slider("Core thickness", 0.2, 1.0, 0.5)
core_l = st.slider("Core length", 5.0, 30.0, 15.0)

col_b = st.slider("Column size", 0.3, 2.0, 0.8)

E = st.number_input("Elastic modulus", value=30e9)

core = CoreWall(core_t, core_l, core_l)
col = Column(col_b, col_b)
tower = Tower(n, h, mass, core, col, E)

if st.button("Run Analysis"):

    T, phi = analyze(tower)
    u, dr = drift(tower)

    st.subheader("Periods")
    st.write(T[:5])

    st.subheader("Max Drift")
    st.write(np.max(np.abs(dr)))

    fig, ax = plt.subplots()
    ax.plot(u, np.arange(1, n+1)*h)
    ax.set_title("Displacement")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(dr, np.arange(1, n)*h)
    ax2.set_title("Drift")
    st.pyplot(fig2)
