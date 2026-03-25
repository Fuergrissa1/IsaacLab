# Nacelle / rotor actuators (Implicit PD)

Configuration: [`tiltrotor_6d.py`](source/tiltrotor_6d_lab/tiltrotor_6d_lab/assets/tiltrotor_6d.py).

## Nacelles (`ImplicitActuatorCfg`)

- **Kp = 40 N·m/rad**, **Kd = 4 N·m·s/rad** (per Isaac Lab implicit joint drive).
- Drive torque scales roughly as **τ ≈ Kp (q_cmd − q) − Kd q̇** (see Isaac Lab docs for exact PhysX form).

### Order-of-magnitude inertia about tilt axis

URDF inertias are given in link frames (not necessarily about the hinge axis). Ballpark for one side:

- Nacelle link: mass ~3.2 kg, diagonal inertias ~0.03–0.03 kg·m² scale.
- Rotor link: mass ~1.65 kg, **Izz ~0.052 kg·m²** (axis-dependent).

A conservative **combined** effective **I_axis ~ 0.05–0.10 kg·m²** for scaling (validate from sim articulation DOF data if needed).

### Quasi-static torque / angular acceleration

- **|τ| ~ Kp |Δq|** for moderate rates (before velocity damping dominates).
- Example: **|Δq| = 0.5 rad** → **|τ| ~ 20 N·m** → **|α| = |τ|/I ~ 200–400 rad/s²** if **I ~ 0.05–0.10 kg·m²**.

This is an upper-bound scale; PhysX substeps, limits, and coupling reduce realized acceleration.

### Characteristic frequency / damping (linearized)

- **ω_n ≈ √(Kp/I)** → with **Kp=40**, **I=0.08** → **ω_n ≈ 22 rad/s** (~3.5 Hz).
- **ζ ≈ Kd / (2√(Kp I))** → **ζ ≈ 4 / (2√3.2) ≈ 1.1** (overdamped-ish at this ballpark).

Control step: **env step** = **decimation × sim.dt** = **4 × (1/120) s ≈ 33 ms**. Nacelle tracking should be compared to this when tuning **Kp/Kd**.

## Rotors

- **Stiffness 0**, **damping 0.1**, **velocity targets** from RPM.
- Angular acceleration of the rotor spin is **not** set by this PD; spin dynamics follow velocity integration and thrust loads.

## Tuning guidance

- If nacelles **lag** commands relative to thrust changes, **increase Kp** slightly and **scale Kd** to keep damping.
- If **chatter** appears, reduce **Kp** or increase **Kd** modestly.
