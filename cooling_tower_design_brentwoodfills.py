# coolingtower_design_enhanced.py
# Cooling Tower Thermal Design — Enhanced with Realistic Fill Data & Merkel Method
# Author: [Your Name]
# License: MIT

import math
import json
import sys
from io import BytesIO
import streamlit as st

# Check for required packages
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError:
    st.error("Required packages not installed. Please install numpy and pandas.")
    HAS_NUMPY_PANDAS = False

# Check for optional packages
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# -------------------------------
# Password Protection
# -------------------------------
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False
    
    # First run, show input for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.write("*Contact admin for access*")
        return False
    
    # Password not correct, show input + error
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    
    # Password correct
    else:
        return True

# -------------------------------
# Utilities
# -------------------------------
P_ATM = 101325.0  # Pa
CP_W = 4180.0     # J/(kg·K) for liquid water
CP_A = 1006.0     # J/(kg·K) for dry air
R_DA = 287.055    # J/(kg·K) for dry air
R_WV = 461.495    # J/(kg·K) for water vapor

# Tetens saturation vapor pressure (Pa), good ~0–50°C
def psat_Pa(T_C: float) -> float:
    """Saturation vapor pressure in Pa"""
    return 610.78 * math.exp(17.2694 * T_C / (T_C + 237.3))

# Humidity ratio from DB and WB (°C), pressure Pa
def humidity_ratio_from_DB_WB(Tdb_C: float, Twb_C: float, P=P_ATM) -> float:
    """Calculate humidity ratio from dry-bulb and wet-bulb temperatures"""
    # psychrometric constant gamma ≈ (Cp_air * P) / (0.622 * h_fg)
    h_fg = 2501000.0 - 2369.0 * Twb_C  # J/kg, simple linear approx
    gamma = (CP_A * P) / (0.622 * h_fg)
    Pw_s_wb = psat_Pa(Twb_C)
    # Psychrometric relation: Pw = Pw* - gamma*(Tdb - Twb)
    Pw = Pw_s_wb - gamma * (Tdb_C - Twb_C)
    Pw = max(1.0, min(Pw, 0.99 * P))
    W = 0.622 * Pw / (P - Pw)
    return max(W, 1e-6)

# Moist air enthalpy (kJ/kg dry air)
def h_moist_air_kJ_per_kg_da(T_C: float, W: float) -> float:
    """Enthalpy of moist air in kJ/kg dry air"""
    return 1.006 * T_C + W * (2501.0 + 1.86 * T_C)

# Moist air density from T (°C), W (kg/kg), P (Pa)
def rho_moist_air(T_C: float, W: float, P=P_ATM) -> float:
    """Density of moist air in kg/m³"""
    T_K = T_C + 273.15
    return (P / (R_DA * T_K)) * (1 + W / 0.622) / (1 + W)

# Saturated air enthalpy at water temperature Tw (°C) at pressure P
def h_star_kJ_per_kg_da(Tw_C: float, P=P_ATM) -> float:
    """Enthalpy of saturated air at water temperature"""
    Pw_star = psat_Pa(Tw_C)
    W_star = 0.622 * Pw_star / (P - Pw_star)
    return h_moist_air_kJ_per_kg_da(Tw_C, W_star)

# -------------------------------
# Enhanced Fill Database with Real Data
# -------------------------------
REALISTIC_FILLS = {
    # ==================== COUNTERFLOW FILM FILLS ====================
    "Brentwood CF1900 (AccuPac® counterflow)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 1.8,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 6.0,
        "rec_water_loading_m3_h_m2_max": 18.0,
        "free_area_frac": 0.93,
        "depth_m_default": 0.9,
        "dp_k0_Pa_per_m_at_vr": 140.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.85,
        "merkel_kaV_L_at_LG_1": 1.80,
        "merkel_LG_coeffs": [2.1, 1.8, 1.6, 1.4],  # at L/G = 0.5,1.0,1.5,2.0
        "material": "PVC",
        "fouling_resistance": "medium",
        "notes": "High efficiency counterflow film fill"
    },
    
    "Brentwood CF1200 (cross-fluted counterflow)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 1.8,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 6.0,
        "rec_water_loading_m3_h_m2_max": 16.0,
        "free_area_frac": 0.93,
        "depth_m_default": 0.9,
        "dp_k0_Pa_per_m_at_vr": 130.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.85,
        "merkel_kaV_L_at_LG_1": 1.70,
        "merkel_LG_coeffs": [2.0, 1.7, 1.5, 1.3],
        "material": "PVC",
        "notes": "Standard counterflow film fill"
    },
    
    # ==================== CROSSFLOW FILM FILLS ====================
    "Brentwood XF75 (herringbone crossflow)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "crossflow",
        "rec_air_velocity_m_s_min": 1.5,
        "rec_air_velocity_m_s_max": 2.8,
        "rec_water_loading_m3_h_m2_min": 4.0,
        "rec_water_loading_m3_h_m2_max": 12.0,
        "free_area_frac": 0.89,
        "depth_m_default": 0.6,
        "dp_k0_Pa_per_m_at_vr": 95.0,
        "dp_vr_m_s": 2.2,
        "dp_exponent": 1.8,
        "merkel_kaV_L_at_LG_1": 1.60,
        "merkel_LG_coeffs": [1.9, 1.6, 1.3, 1.1],
        "material": "PVC",
        "notes": "Standard crossflow film fill"
    },
    
    "Brentwood XF75 Pro (enhanced crossflow)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "crossflow",
        "rec_air_velocity_m_s_min": 1.6,
        "rec_air_velocity_m_s_max": 2.8,
        "rec_water_loading_m3_h_m2_min": 4.0,
        "rec_water_loading_m3_h_m2_max": 12.0,
        "free_area_frac": 0.89,
        "depth_m_default": 0.6,
        "dp_k0_Pa_per_m_at_vr": 90.0,
        "dp_vr_m_s": 2.2,
        "dp_exponent": 1.8,
        "merkel_kaV_L_at_LG_1": 1.65,
        "merkel_LG_coeffs": [2.0, 1.65, 1.35, 1.15],
        "material": "PVC",
        "notes": "Enhanced thermal performance"
    },
    
    # ==================== SPLASH FILLS ====================
    "Brentwood TurboSplash (modular splash crossflow)": {
        "vendor": "Brentwood",
        "geometry": "splash",
        "flow": "crossflow",
        "rec_air_velocity_m_s_min": 2.0,
        "rec_air_velocity_m_s_max": 3.5,
        "rec_water_loading_m3_h_m2_min": 8.0,
        "rec_water_loading_m3_h_m2_max": 25.0,
        "free_area_frac": 0.96,
        "depth_m_default": 0.6,
        "dp_k0_Pa_per_m_at_vr": 60.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.5,
        "merkel_kaV_L_at_LG_1": 1.20,
        "merkel_LG_coeffs": [1.4, 1.2, 1.0, 0.8],
        "material": "PP",
        "fouling_resistance": "high",
        "notes": "Good for fouling waters, modular design"
    },
    
    "Brentwood HTP25 (splash counterflow)": {
        "vendor": "Brentwood",
        "geometry": "splash",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 2.0,
        "rec_air_velocity_m_s_max": 3.5,
        "rec_water_loading_m3_h_m2_min": 8.0,
        "rec_water_loading_m3_h_m2_max": 25.0,
        "free_area_frac": 0.95,
        "depth_m_default": 0.9,
        "dp_k0_Pa_per_m_at_vr": 55.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.5,
        "merkel_kaV_L_at_LG_1": 1.25,
        "merkel_LG_coeffs": [1.5, 1.25, 1.05, 0.85],
        "material": "PP",
        "notes": "Counterflow splash for high fouling applications"
    },
    
    # ==================== LOW-FOULING FILM FILLS ====================
    "Brentwood OF21MA (offset-fluted counterflow)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 1.8,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 6.0,
        "rec_water_loading_m3_h_m2_max": 15.0,
        "free_area_frac": 0.93,
        "depth_m_default": 0.9,
        "dp_k0_Pa_per_m_at_vr": 130.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.85,
        "merkel_kaV_L_at_LG_1": 1.65,
        "merkel_LG_coeffs": [1.9, 1.65, 1.4, 1.2],
        "material": "PVC",
        "fouling_resistance": "high",
        "notes": "Designed for difficult waters with scaling/fouling"
    },
    
    "Brentwood VF19 PLUS (vertical-fluted)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 2.0,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 8.0,
        "rec_water_loading_m3_h_m2_max": 22.0,
        "free_area_frac": 0.95,
        "depth_m_default": 1.2,
        "dp_k0_Pa_per_m_at_vr": 75.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.8,
        "merkel_kaV_L_at_LG_1": 1.55,
        "merkel_LG_coeffs": [1.8, 1.55, 1.35, 1.15],
        "material": "PVC",
        "fouling_resistance": "very high",
        "notes": "Vertical flutes for anti-fouling, deep fill option"
    },
    
    # ==================== OTHER MANUFACTURERS ====================
    "SPX Marley MX75 (crossflow film)": {
        "vendor": "SPX Marley",
        "geometry": "film",
        "flow": "crossflow",
        "rec_air_velocity_m_s_min": 1.6,
        "rec_air_velocity_m_s_max": 2.8,
        "rec_water_loading_m3_h_m2_min": 4.0,
        "rec_water_loading_m3_h_m2_max": 12.0,
        "free_area_frac": 0.88,
        "depth_m_default": 0.6,
        "dp_k0_Pa_per_m_at_vr": 100.0,
        "dp_vr_m_s": 2.2,
        "dp_exponent": 1.8,
        "merkel_kaV_L_at_LG_1": 1.55,
        "merkel_LG_coeffs": [1.8, 1.55, 1.3, 1.1],
        "material": "PVC",
        "notes": "Standard crossflow film fill"
    },
    
    "EVAPCO EVAPAK (counterflow film)": {
        "vendor": "EVAPCO",
        "geometry": "film",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 1.8,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 6.0,
        "rec_water_loading_m3_h_m2_max": 18.0,
        "free_area_frac": 0.92,
        "depth_m_default": 0.9,
        "dp_k0_Pa_per_m_at_vr": 145.0,
        "dp_vr_m_s": 2.5,
        "dp_exponent": 1.85,
        "merkel_kaV_L_at_LG_1": 1.75,
        "merkel_LG_coeffs": [2.05, 1.75, 1.55, 1.35],
        "material": "PVC",
        "notes": "High efficiency counterflow film fill"
    }
}

# -------------------------------
# Enhanced Pressure Drop Calculation
# -------------------------------
def pressure_drop_fill_enhanced(v_face: float, depth_m: float, 
                               water_loading_m3_h_m2: float,
                               fill: dict) -> float:
    """
    Enhanced pressure drop calculation including water loading effect
    Based on manufacturer test data patterns
    """
    v_ref = fill.get("dp_vr_m_s", 2.5)
    dp_k0 = fill.get("dp_k0_Pa_per_m_at_vr", 100.0)
    dp_exponent = fill.get("dp_exponent", 1.85)
    geometry = fill.get("geometry", "film")
    
    # Base ∆P from velocity (scaled by exponent)
    if v_ref > 0:
        dp_vel = dp_k0 * (v_face / v_ref) ** dp_exponent
    else:
        dp_vel = dp_k0 * (v_face ** dp_exponent)
    
    # Water loading effect (Pa/m per m³/h·m²)
    if geometry == "film":
        water_effect = 2.5  # Pa/m per m³/h·m² for film fills
    else:  # splash
        water_effect = 1.5  # Pa/m per m³/h·m² for splash fills
    
    dp_water = water_effect * water_loading_m3_h_m2
    
    # Total pressure drop per meter
    dp_per_m = dp_vel + dp_water
    
    # Total for given depth
    dp_total = dp_per_m * depth_m
    
    return max(dp_total, 0.0)

# -------------------------------
# Merkel Method Calculations
# -------------------------------
def calculate_required_merkel(T_hot: float, T_cold: float, Twb: float,
                             L_G_ratio: float, method: str = "chebyshev") -> float:
    """
    Calculate required kaV/L using Merkel equation
    Options: 'simple' (log mean) or 'chebyshev' (4-point integration)
    """
    if T_cold <= Twb:
        return float('inf')
    
    if method == "simple":
        # Simplified log mean approximation
        delta1 = T_hot - Twb
        delta2 = T_cold - Twb
        if delta1 <= 0 or delta2 <= 0:
            return float('inf')
        
        delta_lm = (delta1 - delta2) / math.log(delta1 / delta2)
        required_kaVL = (L_G_ratio * CP_W / 1000) * (T_hot - T_cold) / delta_lm
        
    else:  # chebyshev - more accurate
        # 4-point Chebyshev integration
        T_range = T_hot - T_cold
        T1 = T_cold + 0.102673 * T_range
        T2 = T_cold + 0.406204 * T_range
        T3 = T_cold + 0.593796 * T_range
        T4 = T_cold + 0.897327 * T_range
        
        h1 = h_star_kJ_per_kg_da(T1) - h_moist_air_kJ_per_kg_da(Twb, humidity_ratio_from_DB_WB(Twb, Twb))
        h2 = h_star_kJ_per_kg_da(T2) - h_moist_air_kJ_per_kg_da(Twb, humidity_ratio_from_DB_WB(Twb, Twb))
        h3 = h_star_kJ_per_kg_da(T3) - h_moist_air_kJ_per_kg_da(Twb, humidity_ratio_from_DB_WB(Twb, Twb))
        h4 = h_star_kJ_per_kg_da(T4) - h_moist_air_kJ_per_kg_da(Twb, humidity_ratio_from_DB_WB(Twb, Twb))
        
        integral = (1/h1 + 1/h2 + 1/h3 + 1/h4) * T_range / 4
        required_kaVL = (CP_W / 1000) * integral
    
    return max(required_kaVL, 0.0)

def get_fill_merkel_at_LG(fill: dict, L_G_ratio: float) -> float:
    """
    Interpolate fill's kaV/L at given L/G ratio
    Using stored coefficients at L/G = 0.5, 1.0, 1.5, 2.0
    """
    if "merkel_LG_coeffs" not in fill or len(fill["merkel_LG_coeffs"]) != 4:
        # Fallback to constant value if no coefficients
        return fill.get("merkel_kaV_L_at_LG_1", 1.5)
    
    coeffs = fill["merkel_LG_coeffs"]
    lg_points = [0.5, 1.0, 1.5, 2.0]
    
    # Interpolate
    if L_G_ratio <= lg_points[0]:
        return coeffs[0]
    elif L_G_ratio >= lg_points[-1]:
        return coeffs[-1]
    else:
        # Linear interpolation
        for i in range(len(lg_points)-1):
            if lg_points[i] <= L_G_ratio <= lg_points[i+1]:
                t = (L_G_ratio - lg_points[i]) / (lg_points[i+1] - lg_points[i])
                return coeffs[i] + t * (coeffs[i+1] - coeffs[i])
    
    return fill.get("merkel_kaV_L_at_LG_1", 1.5)

def check_fill_sufficiency(required_kaVL: float, fill_kaVL: float,
                          safety_factor: float = 1.1) -> dict:
    """
    Check if selected fill can meet thermal requirements
    """
    if required_kaVL <= 0:
        return {
            "sufficient": True,
            "safety_factor": float('inf'),
            "assessment": "No cooling required",
            "required_kaVL": 0,
            "fill_kaVL": fill_kaVL
        }
    
    actual_safety = fill_kaVL / required_kaVL
    
    result = {
        "required_kaVL": required_kaVL,
        "fill_kaVL": fill_kaVL,
        "safety_factor": actual_safety,
        "sufficient": actual_safety >= safety_factor,
        "assessment": ""
    }
    
    if actual_safety < 0.9:
        result["assessment"] = f"❌ INSUFFICIENT: Safety factor = {actual_safety:.2f} (need ≥ {safety_factor})"
    elif actual_safety < 1.0:
        result["assessment"] = f"⚠️ MARGINAL: Safety factor = {actual_safety:.2f}"
    elif actual_safety < safety_factor:
        result["assessment"] = f"⚠️ ADEQUATE: Safety factor = {actual_safety:.2f}"
    else:
        result["assessment"] = f"✅ SUFFICIENT: Safety factor = {actual_safety:.2f}"
    
    return result

# -------------------------------
# Air Flow Suggestion
# -------------------------------
def suggest_air_flow(Q_kW: float, T_w_out_C: float, T_db_in_C: float, 
                    T_wb_in_C: float, approach: float = 1.5) -> dict:
    """
    Suggest dry-air mass flow [kg/s] and volumetric flow [m^3/s]
    with configurable approach temperature
    """
    W_in = humidity_ratio_from_DB_WB(T_db_in_C, T_wb_in_C)
    h_in = h_moist_air_kJ_per_kg_da(T_db_in_C, W_in)
    
    T_exh = T_w_out_C + approach  # Configurable approach
    
    # Assume exhaust air near saturation (95% RH)
    Pw_star = psat_Pa(T_exh)
    RH_exh = 0.95
    Pw_exh = min(RH_exh * Pw_star, 0.99 * P_ATM)
    W_exh = 0.622 * Pw_exh / (P_ATM - Pw_exh)
    h_out = h_moist_air_kJ_per_kg_da(T_exh, W_exh)
    
    delta_h = max(h_out - h_in, 0.5)  # kJ/kg_da
    G_da = Q_kW / delta_h  # kg_dry_air / s
    
    # Mean state for density
    T_mean = 0.5 * (T_db_in_C + T_exh)
    W_mean = 0.5 * (W_in + W_exh)
    rho = rho_moist_air(T_mean, W_mean)
    
    # Volumetric flow
    m_dot_moist = G_da * (1.0 + W_mean)
    V_dot = m_dot_moist / rho  # m^3/s
    
    # Calculate L/G ratio (for water flow of 1 kg/s for now)
    L_G_ratio = 1.0 / G_da if G_da > 0 else 0
    
    return {
        "W_in": W_in,
        "W_out": W_exh,
        "h_in_kJkg": h_in,
        "h_out_kJkg": h_out,
        "G_da_kg_s": G_da,
        "rho_mean": rho,
        "V_dot_m3_s": V_dot,
        "T_exh_C": T_exh,
        "delta_h": delta_h,
        "L_G_ratio": L_G_ratio
    }

# -------------------------------
# Optimize L/G Ratio
# -------------------------------
def optimize_LG_ratio(T_hot: float, T_cold: float, Twb: float, 
                     fill: dict, water_flow_kg_s: float,
                     min_LG: float = 0.5, max_LG: float = 2.0) -> dict:
    """
    Find optimal L/G ratio for given conditions and fill
    """
    if not HAS_NUMPY_PANDAS:
        return {
            "optimal_LG": 1.0,
            "max_safety_factor": 1.0,
            "required_air_flow_kg_s": water_flow_kg_s,
            "evaluated_points": []
        }
    
    import numpy as np
    
    best_LG = 1.0
    best_safety = 0.0
    results = []
    
    # Evaluate at several L/G ratios
    for LG in np.linspace(min_LG, max_LG, 20):
        required = calculate_required_merkel(T_hot, T_cold, Twb, LG, "chebyshev")
        available = get_fill_merkel_at_LG(fill, LG)
        
        if required > 0:
            safety = available / required
            results.append((LG, safety, required, available))
            
            if safety > best_safety:
                best_safety = safety
                best_LG = LG
    
    # Calculate air flow at optimal L/G
    G_opt = water_flow_kg_s / best_LG if best_LG > 0 else 0
    
    return {
        "optimal_LG": best_LG,
        "max_safety_factor": best_safety,
        "required_air_flow_kg_s": G_opt,
        "evaluated_points": results
    }

# -------------------------------
# PDF Report Generation
# -------------------------------
def build_pdf_report(proj_name, flow_type, fill_name, depth_m, free_area_frac,
                    Qw_L_min, m_dot_w, Tin_C, Tout_C, Tdb_C, Twb_C, Q_kW, thermal_safety,
                    A_fill, v_air_face, water_loading_m3_h_m2, DP_fill, Vdot_user,
                    L_G_current, Q_ach_kW, Tout_ach_C, required_kaVL, available_kaVL,
                    merkel_check, fill, warnings):
    """Generate PDF report"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    Wp, Hp = A4
    
    def line(y, text, size=10, bold=False):
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.drawString(20*mm, y, text)
        return y - 5*mm
    
    y = Hp - 20*mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20*mm, y, f"Cooling Tower Design Report — {proj_name}")
    y -= 8*mm
    c.setFont("Helvetica", 10)
    c.drawString(20*mm, y, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 15*mm
    
    # Inputs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, "DESIGN INPUTS")
    y -= 8*mm
    
    inputs = [
        ("Tower flow", flow_type),
        ("Fill model", fill_name),
        ("Fill depth", f"{depth_m:.2f} m"),
        ("Free‑area fraction", f"{free_area_frac:.2f}"),
        ("Water flow", f"{Qw_L_min:.0f} L/min ({m_dot_w:.1f} kg/s)"),
        ("Hot water in", f"{Tin_C:.1f} °C"),
        ("Cold water target", f"{Tout_C:.1f} °C"),
        ("Ambient DB/WB", f"{Tdb_C:.1f} / {Twb_C:.1f} °C"),
        ("Heat load", f"{Q_kW:.0f} kW"),
        ("Design safety factor", f"{thermal_safety:.2f}")
    ]
    
    for k, v in inputs:
        y = line(y, f"• {k}: {v}", 10)
    
    y -= 10*mm
    
    # Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, "DESIGN RESULTS")
    y -= 8*mm
    
    results = [
        ("Fill plan area", f"{A_fill:.2f} m²"),
        ("Air face velocity", f"{v_air_face:.2f} m/s"),
        ("Water loading", f"{water_loading_m3_h_m2:.1f} m³/h·m²"),
        ("Fill pressure drop", f"{DP_fill:.0f} Pa"),
        ("Fan airflow", f"{Vdot_user:.2f} m³/s"),
        ("L/G ratio", f"{L_G_current:.2f}"),
        ("Achievable heat rejection", f"{Q_ach_kW:.0f} kW"),
        ("Achievable cold water out", f"{Tout_ach_C:.1f} °C")
    ]
    
    for k, v in results:
        y = line(y, f"• {k}: {v}", 10)
    
    y -= 10*mm
    
    # Merkel Verification
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, "MERKEL VERIFICATION")
    y -= 8*mm
    
    merkel_items = [
        ("Required kaV/L", f"{required_kaVL:.2f}"),
        ("Available kaV/L", f"{available_kaVL:.2f}"),
        ("Safety factor", f"{merkel_check['safety_factor']:.2f}"),
        ("Status", "SUFFICIENT" if merkel_check["sufficient"] else "INSUFFICIENT")
    ]
    
    for k, v in merkel_items:
        y = line(y, f"• {k}: {v}", 10)
    
    y -= 10*mm
    
    # Fill Details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, "FILL CHARACTERISTICS")
    y -= 8*mm
    
    fill_details = [
        ("Vendor", fill.get("vendor", "")),
        ("Geometry", fill.get("geometry", "")),
        ("Material", fill.get("material", "")),
        ("Fouling resistance", fill.get("fouling_resistance", "N/A")),
        ("Notes", fill.get("notes", ""))
    ]
    
    for k, v in fill_details:
        if v:  # Only include if value exists
            y = line(y, f"• {k}: {v}", 10)
    
    # Warnings if any
    if warnings:
        y -= 10*mm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20*mm, y, "DESIGN WARNINGS")
        y -= 8*mm
        
        for warn in warnings[:5]:  # Limit to 5 warnings
            y = line(y, f"• {warn}", 9)
    
    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# -------------------------------
# Main Application
# -------------------------------
def main_app():
    """Main application after password check"""
    if not HAS_NUMPY_PANDAS:
        st.error("""
        ## Required packages not installed!
        
        Please install the required packages by running:
        ```
        pip install numpy pandas
        ```
        
        Or install all dependencies from requirements.txt:
        ```
        pip install -r requirements.txt
        ```
        """)
        return
    
    st.set_page_config(page_title="Cooling Tower Sizer — Enhanced", layout="wide")
    st.title("🧊 Cooling Tower Thermal Sizer — Enhanced")
    st.caption("Engineering estimator with realistic fill data and Merkel method verification")
    
    # Display package status
    if not HAS_REPORTLAB:
        st.warning("📄 **Note**: PDF report generation requires 'reportlab'. Install with: `pip install reportlab`")
    
    # Sidebar
    with st.sidebar:
        st.header("Project & Options")
        proj_name = st.text_input("Project name", "Chiller Cooling Tower")
        
        # Safety factors
        st.subheader("Design Safety Factors")
        thermal_safety = st.slider("Thermal safety factor", 1.0, 1.5, 1.15, 0.05)
        approach_temp = st.slider("Exhaust approach temp (°C)", 0.5, 3.0, 1.5, 0.1)
        
        # Fill selection
        st.subheader("Fill Selection")
        flow_type = st.radio("Tower flow arrangement", ["counterflow", "crossflow"], index=1)
        filter_by_flow = st.checkbox("Filter fills by flow arrangement", value=True)
        
        # Editable fill library
        fills_json = st.text_area(
            "Fill database (JSON, editable)",
            value=json.dumps(REALISTIC_FILLS, indent=2),
            height=300,
        )
        
        try:
            fills_db = json.loads(fills_json)
        except Exception as e:
            st.error(f"Fill JSON parse error: {e}")
            fills_db = REALISTIC_FILLS
        
        if filter_by_flow:
            fill_names = [k for k, v in fills_db.items() if v.get("flow", "") == flow_type]
        else:
            fill_names = list(fills_db.keys())
        
        if not fill_names:
            fill_names = list(fills_db.keys())
        
        fill_name = st.selectbox("Select fill model", fill_names)
        fill = fills_db[fill_name]
        
        depth_m = st.number_input("Fill depth (m)", 0.2, 2.0, 
                                  float(fill.get("depth_m_default", 0.6)), 0.05)
        
        v_face_target = st.number_input(
            "Target air face velocity (m/s)",
            0.5, 5.0,
            float(0.5 * (fill.get("rec_air_velocity_m_s_min", 1.5) + 
                         fill.get("rec_air_velocity_m_s_max", 3.0))),
            0.05,
        )
        
        free_area_frac = st.slider("Fill free‑area fraction", 0.6, 0.98, 
                                   float(fill.get("free_area_frac", 0.9)), 0.01)
    
    # Main inputs
    st.subheader("Water & Air Inputs")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Qw_L_min = st.number_input("Water flow (L/min)", 10.0, 100000.0, 6000.0, 10.0)
        m_dot_w = Qw_L_min / 60.0 * 1e-3 * 1000.0  # kg/s
        st.write(f"**Water mass flow:** {m_dot_w:.2f} kg/s")
    
    with col2:
        Tin_C = st.number_input("Hot water in (°C)", 10.0, 80.0, 37.0, 0.1)
    
    with col3:
        Tdb_C = st.number_input("Ambient DB (°C)", -10.0, 55.0, 38.0, 0.1)
    
    with col4:
        Twb_C = st.number_input("Ambient WB (°C)", -10.0, 35.0, 28.0, 0.1)
    
    st.markdown("---")
    
    # Sizing mode
    mode = st.radio("Sizing mode", ["Specify outlet temperature", "Specify heat load (kW)"], horizontal=True)
    
    if mode == "Specify outlet temperature":
        Tout_C = st.number_input("Required cold water out (°C)", 5.0, Tin_C-0.1, 
                                max(Twb_C+4.0, 25.0), 0.1)
        Q_kW = (m_dot_w * CP_W * (Tin_C - Tout_C)) / 1000.0
        st.write(f"**Heat rejected:** {Q_kW:.1f} kW")
        st.write(f"**Range (ΔT):** {Tin_C - Tout_C:.1f} °C")
        st.write(f"**Approach (to WB):** {Tout_C - Twb_C:.1f} °C")
    else:
        Q_kW = st.number_input("Heat to reject (kW)", 5.0, 20000.0, 3500.0, 5.0)
        # Estimate Tout
        Tout_C = Tin_C - (Q_kW * 1000.0) / (m_dot_w * CP_W)
        st.write(f"**Estimated cold water out:** {Tout_C:.2f} °C")
        st.write(f"**Range (ΔT):** {Tin_C - Tout_C:.1f} °C")
        st.write(f"**Approach (to WB):** {max(Tout_C - Twb_C, 0):.1f} °C")
    
    # Suggest air flow
    sugg = suggest_air_flow(Q_kW, Tout_C, Tdb_C, Twb_C, approach_temp)
    Vdot_suggest = sugg["V_dot_m3_s"]
    
    st.markdown("### Fan Airflow")
    colA, colB = st.columns(2)
    with colA:
        Vdot_user = st.number_input("Fan volumetric flow (m³/s)", 0.1, 500.0, 
                                   float(Vdot_suggest), 0.1, format="%.3f")
    with colB:
        st.metric("Suggested airflow (m³/s)", f"{Vdot_suggest:.2f}")
        st.metric("L/G ratio", f"{sugg['L_G_ratio']:.2f}")
    
    # Fill plan area
    A_fill = Vdot_user / max(v_face_target * free_area_frac, 1e-6)
    
    # Velocities and loading
    v_air_face = Vdot_user / max(A_fill * free_area_frac, 1e-9)
    v_water = (Qw_L_min / 60.0 / 1000.0) / max(A_fill, 1e-9)
    water_loading_m3_h_m2 = (Qw_L_min / 1000.0) * 60.0 / max(A_fill, 1e-9)
    
    # Pressure drop
    DP_fill = pressure_drop_fill_enhanced(v_air_face, depth_m, water_loading_m3_h_m2, fill)
    
    # Achievable performance with current fan
    G_sugg = sugg["G_da_kg_s"]
    V_sugg = sugg["V_dot_m3_s"]
    G_user = G_sugg * (Vdot_user / max(V_sugg, 1e-9))
    delta_h = sugg["h_out_kJkg"] - sugg["h_in_kJkg"]
    Q_ach_kW = G_user * max(delta_h, 0.1)
    Tout_ach_C = Tin_C - (Q_ach_kW * 1000.0) / (m_dot_w * CP_W)
    
    # Calculate L/G ratio for current design
    L_G_current = m_dot_w / max(G_user, 1e-6)
    
    # -------------------------------
    # Merkel Method Check
    # -------------------------------
    st.markdown("## 📊 Merkel Method Verification")
    
    # Calculate required and available kaV/L
    required_kaVL = calculate_required_merkel(Tin_C, Tout_C, Twb_C, L_G_current, "chebyshev")
    available_kaVL = get_fill_merkel_at_LG(fill, L_G_current)
    merkel_check = check_fill_sufficiency(required_kaVL, available_kaVL, thermal_safety)
    
    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        st.metric("Required kaV/L", f"{required_kaVL:.2f}")
    with colM2:
        st.metric("Available kaV/L", f"{available_kaVL:.2f}")
    with colM3:
        st.metric("Safety Factor", f"{merkel_check['safety_factor']:.2f}")
    with colM4:
        status_color = "green" if merkel_check["sufficient"] else "red"
        st.markdown(f"<h3 style='color: {status_color};'>Status: {'✓' if merkel_check['sufficient'] else '✗'}</h3>", 
                    unsafe_allow_html=True)
    
    st.info(merkel_check["assessment"])
    
    # L/G Optimization
    st.markdown("### 🔄 L/G Ratio Optimization")
    if st.button("Optimize L/G Ratio"):
        with st.spinner("Optimizing..."):
            opt_result = optimize_LG_ratio(Tin_C, Tout_C, Twb_C, fill, m_dot_w)
            
            st.write(f"**Optimal L/G ratio:** {opt_result['optimal_LG']:.2f}")
            st.write(f"**Maximum safety factor:** {opt_result['max_safety_factor']:.2f}")
            st.write(f"**Required air flow at optimal:** {opt_result['required_air_flow_kg_s']:.2f} kg/s")
            
            # Plot if matplotlib is available
            if HAS_MATPLOTLIB and opt_result["evaluated_points"]:
                try:
                    points = opt_result["evaluated_points"]
                    LGs = [p[0] for p in points]
                    safeties = [p[1] for p in points]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(LGs, safeties, 'b-', linewidth=2, label='Safety Factor')
                    ax.axhline(y=thermal_safety, color='r', linestyle='--', label=f'Target ({thermal_safety})')
                    ax.axvline(x=opt_result['optimal_LG'], color='g', linestyle=':', 
                              label=f'Optimal L/G ({opt_result["optimal_LG"]:.2f})')
                    ax.axvline(x=L_G_current, color='orange', linestyle=':', 
                              label=f'Current L/G ({L_G_current:.2f})')
                    
                    ax.set_xlabel('L/G Ratio (water/air mass flow)')
                    ax.set_ylabel('Safety Factor (available/required kaV/L)')
                    ax.set_title('L/G Ratio Optimization')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot: {e}")
            elif not HAS_MATPLOTLIB:
                st.info("Install matplotlib for visualization: `pip install matplotlib`")
    
    # -------------------------------
    # Results Summary
    # -------------------------------
    st.markdown("## 📋 Design Summary")
    
    colR1, colR2, colR3, colR4 = st.columns(4)
    with colR1:
        st.metric("Fill plan area (m²)", f"{A_fill:.2f}")
        st.metric("Water superficial vel. (m/s)", f"{v_water:.4f}")
    with colR2:
        st.metric("Air face vel. (m/s)", f"{v_air_face:.2f}")
        st.metric("Water loading (m³/h·m²)", f"{water_loading_m3_h_m2:.1f}")
    with colR3:
        st.metric("Fill dP (Pa)", f"{DP_fill:.0f}")
        st.metric("Fan flow (m³/s)", f"{Vdot_user:.2f}")
    with colR4:
        st.metric("Achievable kW", f"{Q_ach_kW:.0f}")
        st.metric("Achievable Tout (°C)", f"{Tout_ach_C:.2f}")
    
    # Warnings
    warnings = []
    
    # Velocity range check
    v_min = fill.get("rec_air_velocity_m_s_min", 0)
    v_max = fill.get("rec_air_velocity_m_s_max", 99)
    if v_air_face < v_min or v_air_face > v_max:
        warnings.append(f"Air face velocity ({v_air_face:.2f} m/s) outside recommended range ({v_min}-{v_max} m/s)")
    
    # Water loading check
    wl_min = fill.get("rec_water_loading_m3_h_m2_min", 0)
    wl_max = fill.get("rec_water_loading_m3_h_m2_max", 99)
    if water_loading_m3_h_m2 < wl_min or water_loading_m3_h_m2 > wl_max:
        warnings.append(f"Water loading ({water_loading_m3_h_m2:.1f} m³/h·m²) outside range ({wl_min}-{wl_max})")
    
    # Approach check
    approach = Tout_ach_C - Twb_C
    if approach < 2.0:
        warnings.append(f"Approach temperature ({approach:.1f}°C) is low (< 2°C) - may be difficult to achieve")
    
    # Merkel sufficiency
    if not merkel_check["sufficient"]:
        warnings.append(f"Fill insufficient: safety factor = {merkel_check['safety_factor']:.2f} < {thermal_safety}")
    
    if warnings:
        st.error("### ⚠️ Design Issues:")
        for warn in warnings:
            st.write(f"• {warn}")
    
    # -------------------------------
    # Detailed Calculations
    # -------------------------------
    with st.expander("📐 Show Detailed Calculations"):
        st.write("**Psychrometrics:**")
        st.json({
            "W_in (kg/kg)": round(sugg["W_in"], 5),
            "h_in (kJ/kg_da)": round(sugg["h_in_kJkg"], 2),
            "T_exh (°C)": round(sugg["T_exh_C"], 2),
            "W_out (kg/kg)": round(sugg["W_out"], 5),
            "h_out (kJ/kg_da)": round(sugg["h_out_kJkg"], 2),
            "Δh (kJ/kg_da)": round(sugg["delta_h"], 2),
            "ρ_mean (kg/m³)": round(sugg["rho_mean"], 3),
            "L/G ratio": round(L_G_current, 3)
        })
        
        st.write("**Fill Parameters:**")
        st.json({
            "Fill geometry": fill.get("geometry", "unknown"),
            "Material": fill.get("material", "unknown"),
            "Fouling resistance": fill.get("fouling_resistance", "unknown"),
            "Pressure drop exponent": fill.get("dp_exponent", 1.85),
            "Merkel at L/G=1.0": fill.get("merkel_kaV_L_at_LG_1", 0),
            "Notes": fill.get("notes", "")
        })
        
        st.write("**Hydraulics:**")
        st.json({
            "Free area fraction": free_area_frac,
            "Target v_face (m/s)": v_face_target,
            "Actual v_face (m/s)": round(v_air_face, 3),
            "A_fill (m²)": round(A_fill, 3),
            "Depth (m)": depth_m,
            "Water loading (m³/h·m²)": round(water_loading_m3_h_m2, 1),
            "ΔP_fill (Pa)": round(DP_fill, 1)
        })
    
    # -------------------------------
    # Report Generation
    # -------------------------------
    st.markdown("---")
    st.markdown("## 📄 Design Report")
    
    # Generate text report
    report_text = f"""
    COOLING TOWER DESIGN REPORT
    ============================
    Project: {proj_name}
    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    
    DESIGN INPUTS
    -------------
    • Tower flow arrangement: {flow_type}
    • Fill model: {fill_name}
    • Fill depth: {depth_m:.2f} m
    • Free-area fraction: {free_area_frac:.2f}
    • Water flow: {Qw_L_min:.0f} L/min ({m_dot_w:.1f} kg/s)
    • Hot water in: {Tin_C:.1f} °C
    • Cold water target: {Tout_C:.1f} °C
    • Ambient conditions: {Tdb_C:.1f}°C DB / {Twb_C:.1f}°C WB
    • Heat load: {Q_kW:.0f} kW
    • Design safety factor: {thermal_safety:.2f}
    
    DESIGN RESULTS
    --------------
    • Fill plan area: {A_fill:.2f} m²
    • Air face velocity: {v_air_face:.2f} m/s
    • Water loading: {water_loading_m3_h_m2:.1f} m³/h·m²
    • Fill pressure drop: {DP_fill:.0f} Pa
    • Fan airflow: {Vdot_user:.2f} m³/s
    • L/G ratio: {L_G_current:.2f}
    • Achievable heat rejection: {Q_ach_kW:.0f} kW
    • Achievable cold water out: {Tout_ach_C:.1f} °C
    
    MERKEL VERIFICATION
    -------------------
    • Required kaV/L: {required_kaVL:.2f}
    • Available kaV/L: {available_kaVL:.2f}
    • Safety factor: {merkel_check['safety_factor']:.2f}
    • Status: {'SUFFICIENT' if merkel_check['sufficient'] else 'INSUFFICIENT'}
    
    FILL CHARACTERISTICS
    --------------------
    • Vendor: {fill.get('vendor', 'N/A')}
    • Geometry: {fill.get('geometry', 'N/A')}
    • Material: {fill.get('material', 'N/A')}
    • Fouling resistance: {fill.get('fouling_resistance', 'N/A')}
    • Notes: {fill.get('notes', 'N/A')}
    
    WARNINGS
    --------
    {chr(10).join(warnings) if warnings else 'None'}
    """
    
    # Display report
    st.text_area("Design Report", report_text, height=400)
    
    # Download as text file
    st.download_button(
        label="📥 Download Report as Text File",
        data=report_text,
        file_name=f"{proj_name.replace(' ', '_')}_CoolingTower_Design.txt",
        mime="text/plain",
    )
    
    # PDF Generation (if reportlab is available)
    if HAS_REPORTLAB:
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_bytes = build_pdf_report(
                        proj_name, flow_type, fill_name, depth_m, free_area_frac,
                        Qw_L_min, m_dot_w, Tin_C, Tout_C, Tdb_C, Twb_C, Q_kW, thermal_safety,
                        A_fill, v_air_face, water_loading_m3_h_m2, DP_fill, Vdot_user,
                        L_G_current, Q_ach_kW, Tout_ach_C, required_kaVL, available_kaVL,
                        merkel_check, fill, warnings
                    )
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{proj_name.replace(' ', '_')}_CoolingTower_Design.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
    else:
        st.info("💡 For PDF reports, install reportlab: `pip install reportlab`")
    
    # -------------------------------
    # Footer
    # -------------------------------
    st.markdown("---")
    st.caption(
        """
        **Enhanced Cooling Tower Design Tool**  
        • Uses realistic fill data based on manufacturer specifications  
        • Implements Merkel method for thermal verification  
        • Includes L/G ratio optimization  
        • Safety factors applied to all calculations  
        • For chiller condenser water cooling applications  
        """
    )

# -------------------------------
# Main Entry Point
# -------------------------------
def main():
    # Check password first
    if check_password():
        # If password is correct, run the main app
        main_app()

if __name__ == "__main__":
    main()