"""
V6 — Symmetric PID-Lagrangian Constrained PPO (pilot).

Diagnostic comparator to V4. Tests whether the electronics non-convergence
is driven by asymmetric multiplier saturation (mechanism hypothesis) or
by structural testbed properties (action space, depth, etc.).

PID update rule for each multiplier (lam_util, lam_tp):
    error_t    = -g(pi_hat)                       # violation magnitude (positive = violated)
    P_t        = Kp * error_t
    I_t        = I_{t-1} + Ki * error_t * dt      # with anti-windup, see below
    D_t        = Kd * (error_t - error_{t-1}) / dt
    lam_t      = clip( P_t + I_t + D_t , 0 , lam_max )

Anti-windup (back-calculation): if the unclipped output exceeds lam_max,
the integrator stops accumulating in the saturating direction. This is the
critical difference from V4: I can decrease when slack is positive, AND
the integral does not run away when the controller saturates.

Pilot configuration: matches V4's effective integral gain (Ki = V4's eta),
adds P and D damping. If this saturates anyway, the issue is not asymmetry.
"""

import numpy as np


class PIDLagrangianController:
    """One PID controller per constraint multiplier."""

    def __init__(self, Kp, Ki, Kd, lam_init=0.0, lam_max=np.inf, dt=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.lam_max = lam_max
        self.dt = dt
        self.integral = lam_init / Ki if Ki > 0 else 0.0  # initialise I so lam_0 ≈ lam_init
        self.prev_error = 0.0
        self.lam = lam_init

    def update(self, slack):
        """slack = g(pi_hat). Positive when constraint satisfied, negative when violated."""
        error = -slack  # positive when violated

        P = self.Kp * error
        D = self.Kd * (error - self.prev_error) / self.dt

        # Tentative integral update
        I_candidate = self.integral + self.Ki * error * self.dt
        lam_unclipped = P + I_candidate + D

        # Anti-windup: only accumulate integral if the result stays within bounds
        # (or moves toward bounds from saturation)
        if lam_unclipped > self.lam_max and error > 0:
            # Saturated high and still violating — freeze integral
            I_new = self.integral
        elif lam_unclipped < 0 and error < 0:
            # Saturated low and constraint satisfied — freeze integral
            I_new = self.integral
        else:
            I_new = I_candidate

        self.integral = I_new
        self.prev_error = error

        lam_new = P + I_new + D
        self.lam = float(np.clip(lam_new, 0.0, self.lam_max))
        return self.lam


# ============================================================
# Pilot hyperparameters — Option B (matches V4 effective gain)
# ============================================================
# These mirror V4's integral behaviour, with P and D layered on for damping.
# If electronics still saturates with these, reduce Ki by 10x (Option C).

PID_BAKERY = dict(
    Kp_util=5.0,    Ki_util=5.0,    Kd_util=25.0,    lam_util_init=5.0,    lam_util_max=500.0,
    Kp_tp=100.0,    Ki_tp=1000.0,   Kd_tp=500.0,     lam_tp_init=200.0,    lam_tp_max=20000.0,
)

PID_ELECTRONICS = dict(
    Kp_util=5.0,    Ki_util=5.0,    Kd_util=25.0,    lam_util_init=5.0,    lam_util_max=500.0,
    Kp_tp=100.0,    Ki_tp=1000.0,   Kd_tp=500.0,     lam_tp_init=200.0,    lam_tp_max=20000.0,
)


# ============================================================
# Drop-in replacement for V4's multiplier-update block
# ============================================================
# In pilot_constrained_v4_auto.py, find the V4 update block:
#
#     lam_util = np.clip(lam_util + eta_util * max(0, -util_slack), 0, lam_util_max)
#     lam_tp   = np.clip(lam_tp   + eta_tp   * max(0, -tp_slack),   0, lam_tp_max)
#
# Replace with:
#
#     lam_util = pid_util.update(util_slack)
#     lam_tp   = pid_tp.update(tp_slack)
#
# Initialise controllers once at training start:
#
#     cfg = PID_BAKERY  # or PID_ELECTRONICS
#     pid_util = PIDLagrangianController(
#         Kp=cfg["Kp_util"], Ki=cfg["Ki_util"], Kd=cfg["Kd_util"],
#         lam_init=cfg["lam_util_init"], lam_max=cfg["lam_util_max"],
#     )
#     pid_tp = PIDLagrangianController(
#         Kp=cfg["Kp_tp"], Ki=cfg["Ki_tp"], Kd=cfg["Kd_tp"],
#         lam_init=cfg["lam_tp_init"], lam_max=cfg["lam_tp_max"],
#     )
#
# Log lam_util, lam_tp, util_slack, tp_slack to CSV every episode so we can
# produce Figure 3-style trajectories.


if __name__ == "__main__":
    # Quick unit test: does lambda decrease when slack goes positive?
    pid = PIDLagrangianController(Kp=100, Ki=4000, Kd=100, lam_init=200, lam_max=20000)
    print("=== Sanity check: PID responds symmetrically ===")
    print(f"{'step':>5s} {'slack':>10s} {'P':>10s} {'I':>12s} {'D':>10s} {'lambda':>12s}")
    test_slacks = [-0.02, -0.02, -0.01, 0.0, 0.01, 0.02, 0.02, 0.0, -0.01]
    for i, s in enumerate(test_slacks):
        lam = pid.update(s)
        P = pid.Kp * (-s)
        I = pid.integral * pid.Ki
        D = pid.Kd * ((-s) - pid.prev_error) / pid.dt if i > 0 else 0
        print(f"{i:5d} {s:10.4f} {P:10.2f} {I:12.2f} {D:10.2f} {lam:12.2f}")
    print()
    print("Expected: lambda rises while slack negative, FALLS while slack positive,")
    print("and does not run away even with sustained violation.")
