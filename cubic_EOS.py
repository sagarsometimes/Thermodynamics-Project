import math

# Universal Gas Constants
R = 83.14    # cm3 * bar / (mol * K)  (for PVT calculations)
R_J = 8.314  # J / (mol * K)        (for H and S energy calculations)

# Gas Database (SVNA Table B.1 & C.1)
# Format: "name": [Tc (K), Pc (bar), omega, A, B, C, D]
GASES = {
    "methane":        [190.6, 45.99, 0.012, 1.702, 9.081e-3, -2.164e-6, 0.0],
    "ethane":         [305.3, 48.72, 0.100, 1.131, 19.225e-3, -5.561e-6, 0.0],
    "propane":        [369.8, 42.48, 0.152, 1.213, 28.785e-3, -8.824e-6, 0.0],
    "n-butane":       [425.1, 37.96, 0.200, 1.935, 36.915e-3, -11.402e-6, 0.0],
    "n-pentane":      [469.7, 33.70, 0.252, 2.464, 45.351e-3, -14.111e-6, 0.0],
    "carbon dioxide": [304.2, 73.83, 0.224, 5.457, 1.045e-3, 0.0, -1.157e5],
    "nitrogen":       [126.2, 34.00, 0.038, 3.280, 0.593e-3, 0.0, 0.040e5],
    "water":          [647.1, 220.55, 0.345, 3.470, 1.450e-3, 0.0, 0.121e5],
}


class CubicEOSNewtonRaphson:
    def __init__(self, eos_name):
        self.eos = eos_name.lower()
        
        # Set EOS generic constants (u, w, Omega, Psi)
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
        
        # Kappa parameter for SRK and PR
        if self.eos == "srk":
            k = 0.480 + 1.574 * omega - 0.176 * omega**2
        elif self.eos == "pr":
            k = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            
        return (1 + k * (1 - math.sqrt(Tr))) ** 2

    def dlnalpha_dlnTr(self, Tr, omega):
        if self.eos == "vdw":
            return 0.0
        if self.eos == "rk":
            return -0.5
        
        if self.eos == "srk":
            k = 0.480 + 1.574 * omega - 0.176 * omega**2
        elif self.eos == "pr":
            k = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            
        return -k * math.sqrt(Tr) / (1 + k * (1 - math.sqrt(Tr)))

    def get_a_and_b(self, gas_name, T):
        Tc, Pc, omega = GASES[gas_name][0], GASES[gas_name][1], GASES[gas_name][2]
        Tr = T / Tc
        alpha = self.alpha_function(Tr, omega)
        
        a = self.Psi * alpha * (R**2) * (Tc**2) / Pc
        b = self.Omega * R * Tc / Pc
        return a, b

    def solve_Z(self, gas_name, T, P):
        a, b = self.get_a_and_b(gas_name, T)
        A = a * P / ((R * T) ** 2)
        B = b * P / (R * T)

        # Coefficients for Z^3 + al*Z^2 + be*Z + ga = 0
        al = -1.0 - B + self.u * B
        be = A + self.w * (B**2) - self.u * B - self.u * (B**2)
        ga = -A * B - self.w * (B**2) - self.w * (B**3)

        # --- NEWTON-RAPHSON METHOD ---
        # Initial guess Z0 using Cauchy's root bound (guaranteed to be above all roots)
        Z = 1.0 + abs(al) + abs(be) + abs(ga)

        for _ in range(200):
            # Evaluate cubic function f(Z) and its derivative f'(Z)
            f = Z**3 + al * (Z**2) + be * Z + ga
            f_prime = 3 * (Z**2) + 2 * al * Z + be
            
            # Newton-Raphson update step: Z_new = Z - f(Z)/f'(Z)
            Z_new = Z - f / f_prime
            
            # Check convergence
            if abs(Z_new - Z) < 1e-12:
                break
            Z = Z_new

        # Synthetic division by (Z - Z_new) to find remaining quadratic roots
        Z1 = Z_new
        p = al + Z1
        q = be + Z1 * p
        roots = [Z1]

        disc = p**2 - 4 * q
        if disc >= 0:
            roots.append((-p + math.sqrt(disc)) / 2)
            roots.append((-p - math.sqrt(disc)) / 2)

        # Keep physically valid roots (Z > B) sorted small to large
        valid_roots = sorted([r for r in roots if r > B])
        return valid_roots, A, B

    def calculate_state(self, gas_name, T, P):
        Tc, omega = GASES[gas_name][0], GASES[gas_name][2]
        roots, A, B = self.solve_Z(gas_name, T, P)
        Z = roots[-1]  # Pick vapor root (largest)

        # Sigma and Epsilon parameters
        if self.eos == "pr":
            sigma, eps = 1 + math.sqrt(2), 1 - math.sqrt(2)
        elif self.eos == "vdw":
            sigma, eps = 0.0, 0.0
        else:
            sigma, eps = 1.0, 0.0

        # Density integral I
        if self.eos == "vdw":
            I = B / Z
        else:
            I = math.log((Z + sigma * B) / (Z + eps * B)) / (sigma - eps)

        q = A / B
        d = self.dlnalpha_dlnTr(T / Tc, omega)

        # Calculate residual properties and fugacity
        lnphi = Z - 1.0 - math.log(Z - B) - q * I
        HR = (Z - 1.0 + (d - 1.0) * q * I) * R_J * T
        SR = (math.log(Z - B) + d * q * I) * R_J
        phi = math.exp(lnphi)
        f = phi * P
        V = Z * R * T / P

        return {
            "Z": Z,
            "V": V,
            "HR": HR,
            "SR": SR,
            "phi": phi,
            "f": f,
        }


# --- Benchmark Test Run ---
if __name__ == "__main__":
    gas = "n-butane"
    eos_type = "pr"
    T = 500.0  # K
    P = 50.0   # bar

    solver = CubicEOSNewtonRaphson(eos_type)
    result = solver.calculate_state(gas, T, P)

    print("==================================================")
    print("  Results for", gas, "at", T, "K,", P, "bar (" + eos_type.upper() + " EOS)")
    print("==================================================")
    print("Compressibility Factor (Z) :", round(result["Z"], 5))
    print("Molar Volume (V)          :", round(result["V"], 1), "cm3/mol")
    print("Residual Enthalpy (H^R)   :", round(result["HR"], 0), "J/mol")
    print("Residual Entropy (S^R)    :", round(result["SR"], 3), "J/(mol K)")
    print("Fugacity Coefficient (phi):", round(result["phi"], 4))
    print("Fugacity (f)              :", round(result["f"], 2), "bar")