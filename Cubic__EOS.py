"""
Cubic Equation of State PVT & Fugacity Explorer
Desktop Application (Tkinter + Matplotlib)
Solver: Class-based Newton-Raphson algorithm initialized with Cauchy's bound.
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Universal Gas Constants
R = 83.14    # cm3 * bar / (mol * K)  (PVT calculations)
R_J = 8.314  # J / (mol * K)        (Energy/Entropy calculations)

# Pure Fluid Database (SVNA Tables B.1 and C.1)
# Format: "display_name": [Tc (K), Pc (bar), omega, A, B, C, D]
GASES = {
    "Carbon Dioxide (CO2)": [304.2, 73.83, 0.224, 5.457, 1.045e-3, 0.0, -1.157e5],
    "Methane (CH4)":        [190.6, 45.99, 0.012, 1.702, 9.081e-3, -2.164e-6, 0.0],
    "Ethane (C2H6)":        [305.3, 48.72, 0.100, 1.131, 19.225e-3, -5.561e-6, 0.0],
    "Propane (C3H8)":       [369.8, 42.48, 0.152, 1.213, 28.785e-3, -8.824e-6, 0.0],
    "n-Butane (C4H10)":     [425.1, 37.96, 0.200, 1.935, 36.915e-3, -11.402e-6, 0.0],
    "n-Pentane (C5H12)":    [469.7, 33.70, 0.252, 2.464, 45.351e-3, -14.111e-6, 0.0],
    "Nitrogen (N2)":        [126.2, 34.00, 0.038, 3.280, 0.593e-3, 0.0, 0.040e5],
    "Water (H2O)":          [647.1, 220.55, 0.345, 3.470, 1.450e-3, 0.0, 0.121e5],
}


class CubicEOSNewtonRaphson:
    """Core Thermodynamic Calculation Engine using Newton-Raphson iteration."""

    def __init__(self, eos_name):
        self.eos = eos_name.lower()
        if self.eos == "vdw":
            self.u, self.w, self.Omega, self.Psi = 0, 0, 1 / 8, 27 / 64
        elif self.eos in ["rk", "srk"]:
            self.u, self.w, self.Omega, self.Psi = 1, 0, 0.08664, 0.42748
        elif self.eos == "pr":
            self.u, self.w, self.Omega, self.Psi = 2, -1, 0.07780, 0.45724

    def alpha_function(self, Tr, omega):
        if self.eos == "vdw":
            return 1.0
        if self.eos == "rk":
            return Tr ** (-0.5)
        
        if self.eos == "srk":
            k = 0.480 + 1.574 * omega - 0.176 * omega**2
        elif self.eos == "pr":
            k = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            
        return (1.0 + k * (1.0 - math.sqrt(Tr))) ** 2

    def dlnalpha_dlnTr(self, Tr, omega):
        if self.eos == "vdw":
            return 0.0
        if self.eos == "rk":
            return -0.5
        
        if self.eos == "srk":
            k = 0.480 + 1.574 * omega - 0.176 * omega**2
        elif self.eos == "pr":
            k = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            
        return -k * math.sqrt(Tr) / (1.0 + k * (1.0 - math.sqrt(Tr)))

    def get_a_and_b(self, gas_key, T):
        Tc, Pc, omega = GASES[gas_key][0], GASES[gas_key][1], GASES[gas_key][2]
        Tr = T / Tc
        alpha = self.alpha_function(Tr, omega)
        a = self.Psi * alpha * (R**2) * (Tc**2) / Pc
        b = self.Omega * R * Tc / Pc
        return a, b

    def solve_Z(self, gas_key, T, P):
        a, b = self.get_a_and_b(gas_key, T)
        A = a * P / ((R * T) ** 2)
        B = b * P / (R * T)

        # Coefficients for Z^3 + al*Z^2 + be*Z + ga = 0
        al = -1.0 - B + self.u * B
        be = A + self.w * (B**2) - self.u * B - self.u * (B**2)
        ga = -A * B - self.w * (B**2) - self.w * (B**3)

        # Cauchy's root upper bound for Newton-Raphson initialization
        Z = 1.0 + abs(al) + abs(be) + abs(ga)

        for _ in range(200):
            f = Z**3 + al * (Z**2) + be * Z + ga
            f_prime = 3 * (Z**2) + 2 * al * Z + be
            Z_new = Z - f / f_prime
            if abs(Z_new - Z) < 1e-12:
                break
            Z = Z_new

        # Deflation by synthetic division to find secondary roots
        Z1 = Z_new
        p = al + Z1
        q = be + Z1 * p
        roots = [Z1]

        disc = p**2 - 4 * q
        if disc >= 0:
            roots.append((-p + math.sqrt(disc)) / 2)
            roots.append((-p - math.sqrt(disc)) / 2)

        valid_roots = sorted([r for r in roots if r > B])
        return valid_roots, A, B

    def calculate_state(self, gas_key, T, P):
        Tc, Pc, omega = GASES[gas_key][0], GASES[gas_key][1], GASES[gas_key][2]
        roots, A, B = self.solve_Z(gas_key, T, P)
        
        # Select vapor (largest) or liquid (smallest) root
        Z_vap = roots[-1]
        Z_liq = roots[0]

        if self.eos == "pr":
            sigma, eps = 1 + math.sqrt(2), 1 - math.sqrt(2)
        elif self.eos == "vdw":
            sigma, eps = 0.0, 0.0
        else:
            sigma, eps = 1.0, 0.0

        def evaluate_properties(Z_val):
            if self.eos == "vdw":
                I = B / Z_val
            else:
                I = math.log((Z_val + sigma * B) / (Z_val + eps * B)) / (sigma - eps)

            q = A / B
            d = self.dlnalpha_dlnTr(T / Tc, omega)

            lnphi = Z_val - 1.0 - math.log(Z_val - B) - q * I
            HR = (Z_val - 1.0 + (d - 1.0) * q * I) * R_J * T
            SR = (math.log(Z_val - B) + d * q * I) * R_J
            phi = math.exp(lnphi)
            f_fug = phi * P
            V_m = Z_val * R * T / P
            return Z_val, V_m, HR, SR, phi, f_fug, lnphi

        # Determine thermodynamic phase stability based on Gibbs energy (lnphi)
        Z_v, V_v, HR_v, SR_v, phi_v, f_v, lnphi_v = evaluate_properties(Z_vap)
        
        if len(roots) > 1:
            Z_l, V_l, HR_l, SR_l, phi_l, f_l, lnphi_l = evaluate_properties(Z_liq)
            if lnphi_l < lnphi_v:
                phase_str = "Saturated Liquid"
                selected = (Z_l, V_l, HR_l, SR_l, phi_l, f_l)
            else:
                phase_str = "Saturated Vapor"
                selected = (Z_v, V_v, HR_v, SR_v, phi_v, f_v)
        else:
            if T > Tc:
                phase_str = "Single Phase / Supercritical"
            else:
                phase_str = "Single Phase Fluid"
            selected = (Z_v, V_v, HR_v, SR_v, phi_v, f_v)

        return {
            "Tc": Tc, "Pc": Pc, "omega": omega, "phase": phase_str,
            "Z": selected[0], "V": selected[1], "HR": selected[2],
            "SR": selected[3], "phi": selected[4], "f": selected[5],
            "all_roots": roots
        }


class EOSExplorerApp(tk.Tk):
    """Tkinter Application Window and Interactivity Layout."""

    def __init__(self):
        super().__init__()
        self.title("PVT & Fugacity Real Gas Simulator (Cubic EOS)")
        self.geometry("1100 ...")
        self.minsize(950, 650)

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self._build_header()
        self._build_main_layout()

    def _build_header(self):
        header = tk.Frame(self, bg="#1e293b", height=45)
        header.pack(fill=tk.X, side=tk.TOP)
        lbl = tk.Label(
            header,
            text="Cubic Equation of State PVT & Fugacity Explorer",
            font=("Segoe UI", 13, "bold"),
            fg="#ffffff",
            bg="#1e293b",
            padx=15,
            pady=10
        )
        lbl.pack(side=tk.LEFT)

    def _build_main_layout(self):
        main_frame = tk.Frame(self, bg="#f1f5f9")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel: Controls & Monospace Log
        left_panel = tk.Frame(main_frame, bg="#f8fafc", width=340, highlightbackground="#e2e8f0", highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)

        # Parameter Form
        input_group = ttk.LabelFrame(left_panel, text=" Input Parameters ", padding=10)
        input_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(input_group, text="Substance:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.gas_var = tk.StringVar(value="Carbon Dioxide (CO2)")
        self.gas_combo = ttk.Combobox(input_group, textvariable=self.gas_var, values=list(GASES.keys()), state="readonly", width=20)
        self.gas_combo.grid(row=0, column=1, pady=4, sticky=tk.E)

        ttk.Label(input_group, text="Equation of State:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.eos_var = tk.StringVar(value="PR")
        self.eos_combo = ttk.Combobox(input_group, textvariable=self.eos_var, values=["VDW", "RK", "SRK", "PR"], state="readonly", width=20)
        self.eos_combo.grid(row=1, column=1, pady=4, sticky=tk.E)

        ttk.Label(input_group, text="Temperature T [K]:").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.t_entry = ttk.Entry(input_group, width=22)
        self.t_entry.insert(0, "320.0")
        self.t_entry.grid(row=2, column=1, pady=4, sticky=tk.E)

        ttk.Label(input_group, text="Pressure P [bar]:").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.p_entry = ttk.Entry(input_group, width=22)
        self.p_entry.insert(0, "50.0")
        self.p_entry.grid(row=3, column=1, pady=4, sticky=tk.E)

        calc_btn = ttk.Button(input_group, text="Calculate & Plot", command=self.calculate_and_plot)
        calc_btn.grid(row=4, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Calculated State Log Frame
        log_group = ttk.LabelFrame(left_panel, text=" Calculated Thermodynamic State ", padding=5)
        log_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_group, font=("Consolas", 9), bg="#ffffff", fg="#0f172a", bd=1, relief=tk.SOLID)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Right Panel: Stacked Matplotlib Figures
        right_panel = tk.Frame(main_frame, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        self.fig = Figure(figsize=(6, 6), dpi=96)
        self.ax_pv = self.fig.add_subplot(211)
        self.ax_fug = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Run initial calculation
        self.calculate_and_plot()

    def calculate_and_plot(self):
        # Validate User Inputs
        gas_key = self.gas_var.get()
        eos_name = self.eos_var.get().lower()
        try:
            T = float(self.t_entry.get())
            P = float(self.p_entry.get())
            if T <= 0 or P <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter positive numerical values for T and P.")
            return

        # Solve State
        solver = CubicEOSNewtonRaphson(eos_name)
        state = solver.calculate_state(gas_key, T, P)

        # Update Text Log using String Formatting Specifiers
        self.log_text.delete("1.0", tk.END)
        log_content = (
            f"Fluid : {gas_key.split(' (')[0]}\n"
            f"Model : {eos_name.upper()} | T={T:.1f} K | P={P:.1f} bar\n"
            f"Tc    : {state['Tc']:.1f} K | Pc={state['Pc']:.1f} bar\n"
            f"====================================\n"
            f"[{state['phase']}]\n"
            f"Z        : {state['Z']:.4f}\n"
            f"Vm       : {state['V']:.2f} cm³/mol\n"
            f"phi (φ)  : {state['phi']:.4f}\n"
            f"Fugacity : {state['f']:.3f} bar\n"
            f"H_res    : {state['HR']/1000.0:.2f} kJ/mol\n"
            f"S_res    : {state['SR']:.2f} J/(mol·K)\n"
        )
        self.log_text.insert(tk.END, log_content)

        # Generate Plot Data
        self._update_plots(solver, gas_key, eos_name, T, P, state)

    def _update_plots(self, solver, gas_key, eos_name, T, P, state):
        self.ax_pv.clear()
        self.ax_fug.clear()

        # 1. P-V Isotherm Plot
        v_min = math.log10(max(state['V'] * 0.1, 50.0))
        v_max = math.log10(max(state['V'] * 20.0, 20000.0))
        v_range = [10**x for x in [v_min + i * (v_max - v_min) / 200 for i in range(201)]]
        
        p_isotherm = []
        for v in v_range:
            a, b = solver.get_a_and_b(gas_key, T)
            u, w = solver.u, solver.w
            p_val = (R * T) / (v - b) - a / (v**2 + u * b * v + w * (b**2))
            p_isotherm.append(p_val)

        self.ax_pv.plot(v_range, p_isotherm, color="#1d4ed8", linewidth=1.8, label=f"Isotherm at {T:.1f} K")
        self.ax_pv.axhline(P, color="#ef4444", linestyle=":", linewidth=1.2, label=f"P = {P:.1f} bar")
        self.ax_pv.plot(state['V'], P, marker="o", markersize=6, color="#dc2626")
        
        self.ax_pv.set_xscale("log")
        self.ax_pv.set_title(f"P-V Isotherm ({eos_name.upper()} EOS)", fontsize=11, fontweight="bold")
        self.ax_pv.set_xlabel("Molar Volume Vm [cm³/mol]", fontsize=9)
        self.ax_pv.set_ylabel("Pressure [bar]", fontsize=9)
        self.ax_pv.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        self.ax_pv.legend(loc="upper right", fontsize=8)
        self.ax_pv.set_ylim(bottom=0, top=max(P * 2.5, max(p_isotherm) * 0.5))

        # 2. Pure Gas Fugacity vs Pressure Plot
        p_pressures = [i * (max(P * 2.5, 130.0) / 100) for i in range(1, 101)]
        f_real = []
        for p_i in p_pressures:
            st = solver.calculate_state(gas_key, T, p_i)
            f_real.append(st['f'])

        self.ax_fug.plot(p_pressures, f_real, color="#16a34a", linewidth=1.8, label="EOS Real Gas Fugacity (f)")
        self.ax_fug.plot(p_pressures, p_pressures, color="#64748b", linestyle="--", linewidth=1.2, label="Ideal Gas (f = P)")
        
        self.ax_fug.set_title("Pure Gas Fugacity vs Pressure", fontsize=11, fontweight="bold")
        self.ax_fug.set_xlabel("System Pressure P [bar]", fontsize=9)
        self.ax_fug.set_ylabel("Pure Component Fugacity f [bar]", fontsize=9)
        self.ax_fug.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        self.ax_fug.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


if __name__ == "__main__":
    app = EOSExplorerApp()
    app.mainloop()