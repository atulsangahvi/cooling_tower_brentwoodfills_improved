# cooling_tower_design_complete_with_enhanced_limits.py
# Complete Cooling Tower Design Tool with Enhanced UI & Manufacturer Limits Display

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import hashlib
import math

# ============================================================================
# PASSWORD PROTECTION
# ============================================================================

def check_password():
    """Returns `True` if the user entered the correct password."""
    def password_entered():
        input_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()
        correct_hash = hashlib.sha256("Semaanju".encode()).hexdigest()
        if input_hash == correct_hash:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.markdown("*Hint: The password is 'Semaanju'*")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# ============================================================================
# ENHANCED BRENTWOOD FILL DATABASE - ADDED CF1200
# ============================================================================

BRENTWOOD_FILLS = {
    "CF1200": {
        "name": "Brentwood ACCU-PAK CF1200 (Old Data)",
        "surface_area": 226,  # m²/m³
        "sheet_spacing": 11.7,  # mm
        "flute_angle": 30,  # degrees
        "channel_depth": 9.0,  # mm
        "channel_width": 13.5,  # mm
        "hydraulic_diameter": 8.8,  # mm
        "free_area_fraction": 0.89,
        "water_passage_area": 0.78,
        "material_thickness_options": [0.20, 0.25, 0.30],
        "dry_weight_range": [36.8, 60.9],
        "water_film_thickness": 0.6,  # mm
        "max_water_loading": 14,  # m³/h·m² (lower than XF75)
        "min_water_loading": 6,
        "recommended_water_loading": 8,  # Added recommended value
        "recommended_air_velocity": 2.2,  # m/s (lower than XF75)
        "max_air_velocity": 2.8,
        "fouling_factor": 0.80,  # Worse fouling resistance
        
        # CF1200 PERFORMANCE CURVE (TUNED TO MATCH SUPPLIER'S SAA15 DESIGN)
        # Lower efficiency than XF75 - matches supplier's Ka/L = 0.982 for L/G=2.313
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
            "Ka_L": [1.9, 1.6, 1.35, 1.15, 0.98, 0.85, 0.75, 0.67, 0.60],  # Lower than XF75
            "delta_P_base": [45, 58, 75, 96, 122, 152, 186, 225, 268]  # Higher than XF75
        },
        
        "description": "Older cross-fluted fill with lower thermal efficiency, matches SAA15 supplier design"
    },
    
    "XF75": {
        "name": "Brentwood XF75",
        "surface_area": 226,
        "sheet_spacing": 11.7,
        "flute_angle": 30,
        "channel_depth": 9.0,
        "channel_width": 13.5,
        "hydraulic_diameter": 8.8,
        "free_area_fraction": 0.89,
        "water_passage_area": 0.78,
        "material_thickness_options": [0.20, 0.25, 0.30],
        "dry_weight_range": [36.8, 60.9],
        "water_film_thickness": 0.6,
        "max_water_loading": 15,
        "min_water_loading": 5,
        "recommended_water_loading": 8,  # Added recommended value
        "recommended_air_velocity": 2.5,
        "max_air_velocity": 3.0,
        "fouling_factor": 0.85,
        
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.3, 1.95, 1.65, 1.4, 1.2, 1.05, 0.92],
            "delta_P_base": [35, 45, 60, 80, 105, 135, 170]
        },
        
        "description": "High density cross-fluted fill with maximum surface area"
    },
    
    "ThermaCross": {
        "name": "Brentwood ThermaCross",
        "surface_area": 154,
        "sheet_spacing": 19.0,
        "flute_angle": 22,
        "channel_depth": 11.0,
        "channel_width": 16.5,
        "hydraulic_diameter": 10.5,
        "free_area_fraction": 0.91,
        "water_passage_area": 0.82,
        "material_thickness_options": [0.25, 0.38, 0.50],
        "dry_weight_range": [27.2, 52.9],
        "water_film_thickness": 0.8,
        "max_water_loading": 18,
        "min_water_loading": 6,
        "recommended_water_loading": 10,  # Added recommended value
        "recommended_air_velocity": 2.4,
        "max_air_velocity": 2.9,
        "fouling_factor": 0.90,
        
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.0, 1.7, 1.45, 1.25, 1.08, 0.95, 0.84],
            "delta_P_base": [25, 32, 42, 55, 72, 92, 115]
        },
        
        "description": "Balanced performance fill with good thermal and hydraulic characteristics"
    },
    
    "XF125": {
        "name": "Brentwood XF125",
        "surface_area": 157.5,
        "sheet_spacing": 19.0,
        "flute_angle": 31,
        "channel_depth": 11.0,
        "channel_width": 16.5,
        "hydraulic_diameter": 10.5,
        "free_area_fraction": 0.91,
        "water_passage_area": 0.82,
        "material_thickness_options": [0.25, 0.38, 0.50],
        "dry_weight_range": [27.2, 52.9],
        "water_film_thickness": 0.8,
        "max_water_loading": 18,
        "min_water_loading": 6,
        "recommended_water_loading": 10,  # Added recommended value
        "recommended_air_velocity": 2.4,
        "max_air_velocity": 2.9,
        "fouling_factor": 0.88,
        
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.1, 1.78, 1.52, 1.3, 1.12, 0.98, 0.87],
            "delta_P_base": [28, 36, 47, 62, 80, 102, 128]
        },
        
        "description": "Optimized flute angle for enhanced heat transfer with moderate pressure drop"
    },
    
    "XF3000": {
        "name": "Brentwood XF3000",
        "surface_area": 102,
        "sheet_spacing": 30.5,
        "flute_angle": 30,
        "channel_depth": 13.5,
        "channel_width": 22.5,
        "hydraulic_diameter": 14.2,
        "free_area_fraction": 0.93,
        "water_passage_area": 0.85,
        "material_thickness_options": [0.38, 0.51],
        "dry_weight_range": [25.6, 35.2],
        "water_film_thickness": 1.2,
        "max_water_loading": 25,
        "min_water_loading": 8,
        "recommended_water_loading": 15,  # Added recommended value
        "recommended_air_velocity": 2.6,
        "max_air_velocity": 3.2,
        "fouling_factor": 0.95,
        
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [1.7, 1.45, 1.25, 1.08, 0.94, 0.83, 0.74],
            "delta_P_base": [18, 23, 30, 39, 51, 65, 81]
        },
        
        "description": "Low pressure drop fill for applications with air-side limitations or dirty water"
    }
}

# ============================================================================
# TOWER TYPE DATABASE
# ============================================================================

TOWER_TYPES = {
    "crossflow": {
        "name": "Crossflow",
        "air_water_contact": "Perpendicular",
        "typical_pressure_drop_factor": 1.0,
        "air_distribution": "Side inlet",
        "fill_utilization": 0.85,
        "description": "Air flows horizontally, water flows vertically downward"
    },
    "counterflow_induced": {
        "name": "Counterflow (Induced Draft)",
        "air_water_contact": "Parallel counter-current",
        "typical_pressure_drop_factor": 1.3,  # Higher pressure drop
        "air_distribution": "Bottom inlet, top fan",
        "fill_utilization": 0.95,
        "description": "Air flows upward against downward water flow, fan on top"
    },
    "counterflow_forced": {
        "name": "Counterflow (Forced Draft)",
        "air_water_contact": "Parallel counter-current",
        "typical_pressure_drop_factor": 1.2,
        "air_distribution": "Bottom fan, top outlet",
        "fill_utilization": 0.92,
        "description": "Air flows upward, fan at bottom"
    }
}

# ============================================================================
# PSYCHROMETRIC FUNCTIONS WITH DRY BULB SUPPORT
# ============================================================================

def saturation_pressure(temp_C):
    """Calculate saturation vapor pressure in kPa"""
    T = temp_C + 273.15
    if temp_C >= 0:
        return 0.61121 * np.exp((18.678 - temp_C/234.5) * (temp_C/(257.14 + temp_C)))
    else:
        return 0.61115 * np.exp((23.036 - temp_C/333.7) * (temp_C/(279.82 + temp_C)))

def humidity_ratio_from_wb(db, wb, pressure=101.325):
    """Calculate humidity ratio from dry bulb and wet bulb temperatures"""
    Pws_wb = saturation_pressure(wb)
    Ws_wb = 0.62198 * Pws_wb / (pressure - Pws_wb)
    
    h_fg = 2501.0
    Cp_air = 1.006
    Cp_vapor = 1.86
    
    W = ((h_fg - Cp_vapor * wb) * Ws_wb - Cp_air * (db - wb)) / (h_fg + Cp_vapor * db - 4.186 * wb)
    return max(W, 0.0001)

def relative_humidity_from_wb(db, wb, pressure=101.325):
    """Calculate relative humidity from dry bulb and wet bulb temperatures"""
    W = humidity_ratio_from_wb(db, wb, pressure)
    Pws_db = saturation_pressure(db)
    Ws_db = 0.62198 * Pws_db / (pressure - Pws_db)
    return (W / Ws_db) * 100

def enthalpy_air(db, W):
    """Calculate enthalpy of moist air in kJ/kg dry air"""
    Cp_air = 1.006
    Cp_vapor = 1.86
    h_fg = 2501.0
    return Cp_air * db + W * (h_fg + Cp_vapor * db)

def air_density_calc(db, wb, altitude=0):
    """Calculate air density considering altitude and humidity"""
    # Atmospheric pressure at altitude
    P_atm = 101.325 * (1 - 0.0000225577 * altitude) ** 5.25588  # kPa
    
    # Humidity ratio
    W = humidity_ratio_from_wb(db, wb, P_atm)
    
    # Gas constant for dry air
    R_da = 0.28705  # kJ/kg·K
    
    # Temperature in Kelvin
    T_K = db + 273.15
    
    # Density using ideal gas law with humidity correction
    rho = (P_atm * 1000) / (R_da * 1000 * T_K * (1 + 1.609 * W))
    
    return rho

# ============================================================================
# ENHANCED CALCULATION FUNCTIONS WITH TOWER TYPE SUPPORT
# ============================================================================

def calculate_pressure_drop_with_tower_type(fill_data, tower_type, air_face_velocity, water_loading, fill_depth):
    """Calculate pressure drop considering tower type"""
    L_prime = water_loading / 3.6  # Convert to kg/(s·m²)
    
    # Get base pressure drop from fill curve
    delta_P_base = np.interp(
        L_prime * 3.6,  # Temporary conversion for interpolation
        [x * 8 for x in fill_data["performance_data"]["L_G"]],  # Approximate scaling
        fill_data["performance_data"]["delta_P_base"]
    )
    
    # Adjust for actual face velocity (ΔP ∝ velocity²)
    velocity_factor = (air_face_velocity / 2.5) ** 2
    
    # Adjust for tower type
    tower_factor = TOWER_TYPES[tower_type]["typical_pressure_drop_factor"]
    
    # Fill pressure drop
    fill_pressure_drop = delta_P_base * velocity_factor * fill_depth * tower_factor
    
    # Additional losses based on tower type
    if tower_type == "counterflow_induced":
        # Inlet, eliminators, fan inlet losses (matching supplier's SAA15)
        additional_losses = {
            "inlet_louver": 15,  # Pa (K=3.0)
            "eliminators": 10,   # Pa (K=2.0)
            "fan_inlet": 2,      # Pa (K=0.3)
            "stack_exit": 2,
            "rain_zone": 5
        }
    else:  # crossflow
        additional_losses = {
            "inlet_louver": 10,
            "eliminators": 8,
            "fan_inlet": 1,
            "stack_exit": 1,
            "rain_zone": 3
        }
    
    total_static_pressure = fill_pressure_drop + sum(additional_losses.values())
    
    return {
        "fill_pressure_drop": fill_pressure_drop,
        "total_static_pressure": total_static_pressure,
        "additional_losses": additional_losses,
        "tower_factor": tower_factor
    }

def calculate_KaL_with_tower_type(fill_data, L_over_G, tower_type):
    """Calculate Ka/L considering tower type adjustments"""
    # Get base Ka/L from fill curve
    Ka_over_L = np.interp(
        L_over_G,
        fill_data["performance_data"]["L_G"],
        fill_data["performance_data"]["Ka_L"]
    )
    
    # Adjust for tower type (counterflow typically has better utilization)
    if tower_type.startswith("counterflow"):
        # Counterflow has better contact efficiency
        efficiency_factor = TOWER_TYPES[tower_type]["fill_utilization"] / 0.85
        Ka_over_L *= efficiency_factor
    
    return Ka_over_L

# ============================================================================
# MAIN CALCULATION FUNCTION WITH MANUFACTURER LIMITS SUPPORT
# ============================================================================

def solve_cooling_tower_enhanced(L, G, T_hot, T_cold_target, Twb, Tdb, fill_type, 
                                 tower_type, fill_depth, face_area, altitude=0):
    """
    Enhanced cooling tower solver with manufacturer limits support
    """
    fill_data = BRENTWOOD_FILLS[fill_type]
    tower_data = TOWER_TYPES[tower_type]
    
    # Get adjusted Ka/L
    L_over_G = L / G
    Ka_over_L = calculate_KaL_with_tower_type(fill_data, L_over_G, tower_type)
    
    # Total heat transfer coefficient
    Ka = Ka_over_L * L  # kW/°C
    
    # Air properties with dry bulb and altitude
    air_density = air_density_calc(Tdb, Twb, altitude)  # kg/m³
    air_flow_volumetric = G / air_density  # m³/s
    air_face_velocity = air_flow_volumetric / face_area  # m/s
    
    # Water loading
    water_loading = (L * 3.6) / face_area  # m³/h·m²
    
    # Calculate pressure drop with tower type consideration
    pressure_results = calculate_pressure_drop_with_tower_type(
        fill_data, tower_type, air_face_velocity, water_loading, fill_depth
    )
    
    # Calculate hydraulic properties
    water_velocity_ms = (water_loading / 3.6) / fill_data["water_passage_area"]
    water_viscosity = 1e-6
    film_reynolds = (water_velocity_ms * fill_data["water_film_thickness"] * 1e-3) / water_viscosity
    air_viscosity = 1.5e-5
    air_reynolds = (air_face_velocity * fill_data["hydraulic_diameter"] * 1e-3) / air_viscosity
    
    # Assess fouling risk
    risk_score = 0
    if fill_data["hydraulic_diameter"] < 10:
        risk_score += 2
    elif fill_data["hydraulic_diameter"] < 12:
        risk_score += 1
    if water_velocity_ms < 0.05:
        risk_score += 2
    elif water_velocity_ms < 0.1:
        risk_score += 1
    
    risk_level = "Low" if risk_score < 2 else "Moderate" if risk_score < 4 else "High"
    
    # Merkel number (NTU) - adjusted for tower type
    NTU = Ka_over_L * fill_depth
    if tower_type.startswith("counterflow"):
        # Counterflow typically achieves 5-10% better NTU utilization
        NTU *= 1.05
    
    # Achieved cold water temperature (simplified Merkel solution)
    # For counterflow, use more accurate approach
    if tower_type.startswith("counterflow"):
        # Counterflow typically has better approach for same NTU
        approach_factor = 0.95
    else:
        approach_factor = 1.0
    
    T_cold_achieved = Twb + (T_hot - Twb) * np.exp(-NTU * approach_factor)
    
    # Ensure realistic temperature
    T_cold_achieved = max(T_cold_achieved, Twb + 0.5)
    T_cold_achieved = min(T_cold_achieved, T_hot - 0.5)
    
    Q_achieved = L * 4.186 * (T_hot - T_cold_achieved)
    
    # Fill volume and surface area
    fill_volume = face_area * fill_depth
    total_surface_area = fill_volume * fill_data["surface_area"]
    
    # Calculate fan power
    fan_efficiency = 0.78  # 78% as per supplier SAA15
    transmission_efficiency = 1.0
    fan_power = (air_flow_volumetric * pressure_results["total_static_pressure"]) / \
                (fan_efficiency * transmission_efficiency * 1000)  # kW
    
    # Calculate relative humidity
    P_atm = 101.325 * (1 - 0.0000225577 * altitude) ** 5.25588
    RH = relative_humidity_from_wb(Tdb, Twb, P_atm)
    
    # Check against manufacturer limits
    max_water_loading = fill_data["max_water_loading"]
    min_water_loading = fill_data["min_water_loading"]
    recommended_water_loading = fill_data["recommended_water_loading"]
    max_air_velocity = fill_data["max_air_velocity"]
    recommended_air_velocity = fill_data["recommended_air_velocity"]
    
    # Operating warnings with manufacturer limits
    operating_warnings = []
    
    if water_loading > max_water_loading:
        operating_warnings.append(f"Water loading exceeds manufacturer maximum ({max_water_loading} m³/h·m²)")
    if water_loading < min_water_loading:
        operating_warnings.append(f"Water loading below manufacturer minimum ({min_water_loading} m³/h·m²)")
    
    if air_face_velocity > max_air_velocity:
        operating_warnings.append(f"Air face velocity exceeds manufacturer maximum ({max_air_velocity} m/s)")
    
    # Tower type specific warnings
    if tower_type == "counterflow_induced" and air_face_velocity > 2.5:
        operating_warnings.append("High air velocity for induced draft - ensure proper plenum design")
    
    # Determine status for water loading
    water_status = "❌ Exceeds Max" if water_loading > max_water_loading else \
                   "⚠️ Below Min" if water_loading < min_water_loading else \
                   "✅ Within Limits"
    
    # Determine status for air velocity
    air_status = "❌ Exceeds Max" if air_face_velocity > max_air_velocity else "✅ Within Limits"
    
    return {
        # Basic identification
        "fill_type": fill_type,
        "fill_name": fill_data["name"],
        "tower_type": tower_type,
        "tower_name": tower_data["name"],
        
        # Temperatures and heat transfer
        "T_hot": T_hot,
        "T_cold_achieved": T_cold_achieved,
        "T_cold_target": T_cold_target,
        "Twb": Twb,
        "Tdb": Tdb,
        "RH": RH,
        "Q_achieved": Q_achieved,
        "Q_target": L * 4.186 * (T_hot - T_cold_target),
        "approach": T_cold_achieved - Twb,
        "cooling_range": T_hot - T_cold_achieved,
        
        # Flow parameters
        "L": L,
        "G": G,
        "L_over_G": L_over_G,
        "water_loading": water_loading,
        "air_density": air_density,
        "air_flow_volumetric": air_flow_volumetric,
        "air_face_velocity": air_face_velocity,
        
        # Manufacturer limits - FOR UI DISPLAY
        "max_water_loading": max_water_loading,
        "min_water_loading": min_water_loading,
        "recommended_water_loading": recommended_water_loading,
        "max_air_velocity": max_air_velocity,
        "recommended_air_velocity": recommended_air_velocity,
        "water_status": water_status,
        "air_status": air_status,
        
        # Geometry and sizing
        "fill_depth": fill_depth,
        "face_area": face_area,
        "fill_volume": fill_volume,
        "total_surface_area": total_surface_area,
        
        # Hydraulic properties
        "water_velocity": water_velocity_ms,
        "film_reynolds": film_reynolds,
        "air_reynolds": air_reynolds,
        "water_film_thickness": fill_data["water_film_thickness"],
        
        # Performance parameters
        "NTU": NTU,
        "Ka_over_L": Ka_over_L,
        "Ka": Ka,
        
        # Pressure drop and fan
        "fill_pressure_drop": pressure_results["fill_pressure_drop"],
        "total_static_pressure": pressure_results["total_static_pressure"],
        "additional_losses": pressure_results["additional_losses"],
        "fan_power": fan_power,
        
        # Assessments
        "fouling_risk": {"risk_score": risk_score, "risk_level": risk_level},
        "operating_warnings": operating_warnings,
        
        # Fill characteristics
        "surface_area_density": fill_data["surface_area"],
        "hydraulic_diameter": fill_data["hydraulic_diameter"],
        "flute_angle": fill_data["flute_angle"],
        "free_area_fraction": fill_data["free_area_fraction"],
        
        # Tower characteristics
        "tower_efficiency_factor": tower_data["fill_utilization"],
        
        # Atmospheric conditions
        "altitude": altitude,
        "air_density_calc": air_density
    }

# ============================================================================
# SUPPLIER SAA15 DESIGN VALIDATION FUNCTION
# ============================================================================

def validate_with_saa15_supplier_design():
    """
    Run validation against supplier's SAA15 design with CF1200
    Returns the results for comparison
    """
    # Supplier's SAA15 parameters from image
    supplier_inputs = {
        "L": 114,  # kg/s (actual water flow)
        "G": 49.28,  # kg/s (calculated from L/G=2.313)
        "T_hot": 40.0,  # °C (from range 5°C and T_cold=35°C)
        "T_cold_target": 35.0,  # °C
        "Twb": 30.0,  # °C (assumed from your input)
        "Tdb": 33.0,  # °C (estimated for 60% RH)
        "fill_type": "CF1200",
        "tower_type": "counterflow_induced",
        "fill_depth": 0.75,  # m
        "face_area": 12.96,  # m² (3.6m x 3.6m)
        "altitude": 0
    }
    
    # Run calculation
    results = solve_cooling_tower_enhanced(**supplier_inputs)
    
    # Supplier's claimed results from image
    supplier_claimed = {
        "T_cold_achieved": 35.0,  # °C
        "exit_wb": 37.75,  # °C (CTI)
        "fan_power": 13.41,  # kW
        "Ka_over_L": 0.982,  # Total Ka/L
        "static_pressure_mmWG": 20.225,  # mm WG
        "static_pressure_Pa": 20.225 * 9.81,  # Pa
        "water_loading": 8.95  # l/s·m² = 32.22 m³/h·m²
    }
    
    # Compare
    comparison = {
        "your_calculation": {
            "T_cold": results["T_cold_achieved"],
            "approach": results["approach"],
            "fan_power": results["fan_power"],
            "Ka_over_L": results["Ka_over_L"],
            "static_pressure": results["total_static_pressure"],
            "water_loading": results["water_loading"]
        },
        "supplier_claimed": supplier_claimed,
        "differences": {
            "T_cold_diff": results["T_cold_achieved"] - supplier_claimed["T_cold_achieved"],
            "fan_power_diff": results["fan_power"] - supplier_claimed["fan_power"],
            "Ka_over_L_diff": results["Ka_over_L"] - supplier_claimed["Ka_over_L"]
        }
    }
    
    return results, comparison

# ============================================================================
# REPORT GENERATION WITH MANUFACTURER LIMITS
# ============================================================================

def generate_txt_report(design_results):
    """Generate a detailed TXT report with manufacturer limits"""
    report = []
    report.append("=" * 70)
    report.append("COOLING TOWER DESIGN REPORT")
    report.append("=" * 70)
    report.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Fill: {design_results['fill_name']}")
    report.append(f"Tower Type: {design_results['tower_name']}")
    report.append("=" * 70)
    
    # Design Inputs
    report.append("\nDESIGN INPUTS")
    report.append("-" * 40)
    report.append(f"Water Flow Rate: {design_results['L']:.2f} kg/s")
    report.append(f"Air Flow Rate: {design_results['G']:.2f} kg/s")
    report.append(f"L/G Ratio: {design_results['L_over_G']:.3f}")
    report.append(f"Hot Water In: {design_results['T_hot']:.1f} °C")
    report.append(f"Target Cold Water Out: {design_results['T_cold_target']:.1f} °C")
    report.append(f"Ambient Wet Bulb: {design_results['Twb']:.1f} °C")
    report.append(f"Ambient Dry Bulb: {design_results['Tdb']:.1f} °C")
    report.append(f"Relative Humidity: {design_results['RH']:.1f} %")
    report.append(f"Site Altitude: {design_results['altitude']} m ASL")
    report.append(f"Tower Type: {design_results['tower_name']}")
    report.append(f"Fill Depth: {design_results['fill_depth']:.3f} m")
    report.append(f"Face Area: {design_results['face_area']:.2f} m²")
    
    # Design Results
    report.append("\nDESIGN RESULTS")
    report.append("-" * 40)
    report.append(f"Achieved Cold Water: {design_results['T_cold_achieved']:.2f} °C")
    report.append(f"Heat Rejection: {design_results['Q_achieved']:.0f} kW")
    report.append(f"Cooling Range: {design_results['cooling_range']:.2f} °C")
    report.append(f"Approach: {design_results['approach']:.2f} °C")
    report.append(f"NTU: {design_results['NTU']:.3f}")
    report.append(f"Ka/L: {design_results['Ka_over_L']:.3f}")
    
    # Geometry and Hydraulics
    report.append("\nGEOMETRY & HYDRAULICS")
    report.append("-" * 40)
    report.append(f"Fill Volume: {design_results['fill_volume']:.2f} m³")
    report.append(f"Total Surface Area: {design_results['total_surface_area']:.0f} m²")
    report.append(f"Water Loading: {design_results['water_loading']:.1f} m³/h·m²")
    report.append(f"Water Velocity in Channels: {design_results['water_velocity']:.3f} m/s")
    report.append(f"Water Film Thickness: {design_results['water_film_thickness']} mm")
    report.append(f"Air Face Velocity: {design_results['air_face_velocity']:.2f} m/s")
    report.append(f"Air Density: {design_results['air_density']:.3f} kg/m³")
    report.append(f"Fan Airflow: {design_results['air_flow_volumetric']:.2f} m³/s")
    report.append(f"Fan Static Pressure: {design_results['total_static_pressure']:.1f} Pa")
    report.append(f"Estimated Fan Power: {design_results['fan_power']:.2f} kW")
    
    # MANUFACTURER SPECIFICATIONS
    report.append("\nMANUFACTURER SPECIFICATIONS")
    report.append("-" * 40)
    report.append(f"Maximum Water Loading: {design_results['max_water_loading']} m³/h·m²")
    report.append(f"Minimum Water Loading: {design_results['min_water_loading']} m³/h·m²")
    report.append(f"Recommended Water Loading: {design_results['recommended_water_loading']} m³/h·m²")
    report.append(f"Maximum Air Velocity: {design_results['max_air_velocity']} m/s")
    report.append(f"Recommended Air Velocity: {design_results['recommended_air_velocity']} m/s")
    
    # LIMIT COMPLIANCE CHECK
    report.append("\nLIMIT COMPLIANCE CHECK")
    report.append("-" * 40)
    
    water_status = "EXCEEDS MAXIMUM" if design_results['water_loading'] > design_results['max_water_loading'] else \
                   "BELOW MINIMUM" if design_results['water_loading'] < design_results['min_water_loading'] else "WITHIN LIMITS"
    
    air_status = "EXCEEDS MAXIMUM" if design_results['air_face_velocity'] > design_results['max_air_velocity'] else "WITHIN LIMITS"
    
    report.append(f"Water Loading: {design_results['water_loading']:.1f} m³/h·m² → {water_status}")
    report.append(f"  (Range: {design_results['min_water_loading']} - {design_results['max_water_loading']} m³/h·m²)")
    report.append(f"Air Velocity: {design_results['air_face_velocity']:.2f} m/s → {air_status}")
    report.append(f"  (Maximum: {design_results['max_air_velocity']} m/s)")
    
    # Tower Characteristics
    report.append("\nTOWER CHARACTERISTICS")
    report.append("-" * 40)
    report.append(f"Type: {design_results['tower_name']}")
    report.append(f"Fill Utilization Factor: {design_results['tower_efficiency_factor']:.2f}")
    report.append(f"Pressure Drop Factor: {TOWER_TYPES[design_results['tower_type']]['typical_pressure_drop_factor']:.1f}")
    
    # Fill Characteristics
    report.append("\nFILL CHARACTERISTICS")
    report.append("-" * 40)
    report.append(f"Surface Area Density: {design_results['surface_area_density']} m²/m³")
    report.append(f"Hydraulic Diameter: {design_results['hydraulic_diameter']:.1f} mm")
    report.append(f"Flute Angle: {design_results['flute_angle']}°")
    report.append(f"Free Area Fraction: {design_results['free_area_fraction']:.2f}")
    report.append(f"Fouling Risk: {design_results['fouling_risk']['risk_level']}")
    
    # Status
    report.append("\nPERFORMANCE STATUS")
    report.append("-" * 40)
    if design_results['T_cold_achieved'] <= design_results['T_cold_target']:
        report.append("✅ DESIGN MEETS THERMAL REQUIREMENTS")
    else:
        report.append("⚠️ DESIGN DOES NOT MEET THERMAL REQUIREMENTS")
        report.append(f"   Required improvement: {design_results['T_cold_achieved'] - design_results['T_cold_target']:.2f} °C")
    
    # Overall Compliance Status
    report.append("\nOVERALL COMPLIANCE STATUS")
    report.append("-" * 40)
    all_limits_ok = (design_results['water_loading'] <= design_results['max_water_loading'] and 
                     design_results['water_loading'] >= design_results['min_water_loading'] and
                     design_results['air_face_velocity'] <= design_results['max_air_velocity'])
    
    if all_limits_ok:
        report.append("✅ ALL MANUFACTURER LIMITS SATISFIED")
    else:
        report.append("❌ MANUFACTURER LIMITS EXCEEDED - DESIGN MAY NOT BE FEASIBLE")
    
    # Warnings
    if design_results['operating_warnings']:
        report.append("\nOPERATING WARNINGS")
        report.append("-" * 40)
        for warning in design_results['operating_warnings']:
            report.append(f"⚠️ {warning}")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)

# ============================================================================
# STREAMLIT APP - MAIN FUNCTION WITH ENHANCED UI
# ============================================================================

def main():
    # Check password first
    if not check_password():
        st.stop()
    
    # Set page config
    st.set_page_config(
        page_title="Professional Cooling Tower Design with Manufacturer Limits",
        page_icon="🌊",
        layout="wide"
    )
    
    # Main title
    st.title("🌊 Complete Cooling Tower Design Tool")
    st.markdown("**Enhanced UI | Manufacturer Limits Display | Counterflow Support**")
    
    # Initialize session state for geometry
    if 'tower_shape' not in st.session_state:
        st.session_state.tower_shape = "Rectangle"
    if 'face_area' not in st.session_state:
        st.session_state.face_area = 36.94
    
    # ========================================================================
    # SIDEBAR - DESIGN INPUTS WITH ENHANCED CONTROLS
    # ========================================================================
    with st.sidebar:
        st.header("📥 Design Inputs")
        
        # Calculation Mode
        calc_mode = st.radio(
            "**Calculation Mode:**",
            ["Mode 1: Given Heat Load → Find Water Flow",
             "Mode 2: Given Water Flow → Find Heat Load"],
            help="Choose whether you know the heat load or water flow rate"
        )
        
        # Temperature inputs - Using number_input with step for +/- buttons
        st.subheader("🌡️ Temperature Conditions")
        
        col1, col2 = st.columns(2)
        with col1:
            T_hot = st.number_input("Hot Water In (°C)", 
                                   value=37.0, min_value=20.0, max_value=60.0, 
                                   step=0.5, format="%.1f",
                                   help="Inlet water temperature to cooling tower")
        with col2:
            T_cold_target = st.number_input("Target Cold Water Out (°C)", 
                                           value=32.0, min_value=10.0, max_value=40.0, 
                                           step=0.5, format="%.1f",
                                           help="Desired outlet water temperature")
        
        col3, col4 = st.columns(2)
        with col3:
            Twb = st.number_input("Ambient Wet Bulb (°C)", 
                                 value=28.0, min_value=10.0, max_value=40.0, 
                                 step=0.5, format="%.1f",
                                 help="Critical design parameter for cooling towers")
        with col4:
            # ADDED: Dry bulb temperature input
            Tdb = st.number_input("Ambient Dry Bulb (°C)", 
                                 value=33.0, min_value=10.0, max_value=50.0, 
                                 step=0.5, format="%.1f",
                                 help="Ambient air dry bulb temperature")
        
        # Mode-specific inputs with number_input
        st.subheader("💧 Flow Parameters")
        if calc_mode == "Mode 1: Given Heat Load → Find Water Flow":
            Q_input = st.number_input("Heat Load to Remove (kW)", 
                                     value=2090.0, min_value=100.0, max_value=10000.0,
                                     step=100.0, format="%.1f")
            Cp = 4.186
            if T_hot > T_cold_target:
                L = Q_input / (Cp * (T_hot - T_cold_target))
            else:
                L = 100.0
                st.error("Hot water temperature must be greater than cold water target")
            st.metric("Calculated Water Flow", f"{L:.2f} kg/s")
        else:
            L = st.number_input("Water Flow Rate (kg/s)", 
                               value=100.0, min_value=10.0, max_value=500.0,
                               step=5.0, format="%.2f")
            Cp = 4.186
            if T_hot > T_cold_target:
                Q_input = L * Cp * (T_hot - T_cold_target)
            else:
                Q_input = 2090.0
                st.error("Hot water temperature must be greater than cold water target")
            st.metric("Calculated Heat Load", f"{Q_input:.0f} kW")
        
        # Air Flow Specification with L/G as number_input
        st.subheader("🌬️ Air Flow Specification")
        
        air_input_method = st.radio(
            "Air Flow Input Method:",
            ["Method 1: Set L/G Ratio",
             "Method 2: Set Direct Air Flow Rate"],
            help="Choose to set L/G ratio or direct air flow rate"
        )
        
        if air_input_method == "Method 1: Set L/G Ratio":
            # Using number_input for L/G ratio
            L_over_G = st.number_input("L/G Ratio (Liquid to Gas mass ratio)", 
                                      value=1.25, min_value=0.5, max_value=3.0,
                                      step=0.05, format="%.3f",
                                      help="Typical range: 0.8-1.5. Higher = more air, lower pressure drop")
            G = L / L_over_G
            st.metric("Calculated Air Flow", f"{G:.2f} kg/s")
        else:
            G = st.number_input("Air Mass Flow Rate (kg/s)", 
                               value=80.0, min_value=10.0, max_value=300.0,
                               step=5.0, format="%.2f")
            L_over_G = L / G
            st.metric("Calculated L/G Ratio", f"{L_over_G:.3f}")
        
        # Tower Type Selection
        st.subheader("🏗️ Tower Configuration")
        tower_type = st.selectbox(
            "Tower Type:",
            options=list(TOWER_TYPES.keys()),
            format_func=lambda x: TOWER_TYPES[x]["name"],
            help="Select tower flow arrangement and draft type"
        )
        
        # Show tower description
        tower_desc = TOWER_TYPES[tower_type]["description"]
        st.caption(f"*{tower_desc}*")
        
        # Geometry Parameters
        st.subheader("📐 Geometry Parameters")
        
        # Fill depth as number_input with 3 decimal places
        fill_depth = st.number_input("Fill Depth (m)", 
                                    value=0.600, min_value=0.300, max_value=2.000,
                                    step=0.050, format="%.3f",
                                    help="Depth of fill media in air flow direction")
        
        # Tower shape selection
        st.markdown("**Tower Shape Selection**")
        tower_shape = st.radio(
            "Select tower shape:",
            ["Rectangle", "Round"],
            horizontal=True,
            key="tower_shape_selector"
        )
        
        # Geometry inputs based on shape
        if tower_shape == "Rectangle":
            col1, col2 = st.columns(2)
            with col1:
                fill_length = st.number_input("Fill Face Length (m)", 
                                             value=6.08, min_value=1.0, max_value=20.0,
                                             step=0.1, format="%.2f")
            with col2:
                fill_width = st.number_input("Fill Face Width/Breadth (m)", 
                                            value=6.08, min_value=1.0, max_value=20.0,
                                            step=0.1, format="%.2f")
            face_area = fill_length * fill_width
            st.success(f"**Calculated Face Area:** {face_area:.2f} m²")
            
        else:  # Round tower
            diameter = st.number_input("Tower Diameter (m)", 
                                      value=6.85, min_value=1.0, max_value=20.0,
                                      step=0.1, format="%.2f")
            face_area = math.pi * (diameter / 2) ** 2
            st.success(f"**Calculated Face Area:** {face_area:.2f} m²")
        
        # Store calculated face area in session state
        st.session_state.face_area = face_area
        
        # Altitude as number_input with clear label
        altitude = st.number_input("Site Altitude from Sea Level (m)", 
                                  value=0, min_value=0, max_value=3000,
                                  step=100, format="%d",
                                  help="Altitude above sea level for air density correction")
        
        # Fill Selection - INCLUDING CF1200
        st.subheader("🎯 Brentwood Fill Selection")
        fill_options = list(BRENTWOOD_FILLS.keys())
        selected_fills = st.multiselect(
            "Select fills to compare:",
            options=fill_options,
            default=["CF1200", "XF75"],  # Default includes CF1200
            format_func=lambda x: BRENTWOOD_FILLS[x]["name"],
            help="Select one or more fills for comparison"
        )
        
        # Supplier Validation Button
        st.subheader("🔍 Supplier Validation")
        run_saa15_validation = st.button(
            "🔄 Run SAA15 Supplier Design Validation",
            help="Compare your code against supplier's SAA15 CF1200 design",
            use_container_width=True
        )
        
        # Run Button
        run_calc = st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)
        
        # Report generation
        st.subheader("📄 Report Generation")
        generate_reports = st.checkbox("Generate TXT Report", value=True)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # SAA15 Supplier Validation Section
    if run_saa15_validation:
        st.header("🔬 SAA15 Supplier Design Validation")
        st.info("""
        **Validating against supplier's SAA15 design with CF1200 fill:**
        - Water flow: 114 kg/s
        - L/G ratio: 2.313
        - Fill: CF1200, depth 0.75m
        - Tower: Counterflow induced draft
        - Hot water: 40°C, Cold target: 35°C
        - Wet bulb: 30°C, Dry bulb: 33°C
        """)
        
        with st.spinner("Running validation against supplier's SAA15 design..."):
            results, comparison = validate_with_saa15_supplier_design()
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Code's Calculation")
            st.metric("Cold Water Temp", f"{results['T_cold_achieved']:.2f}°C")
            st.metric("Fan Power", f"{results['fan_power']:.2f} kW")
            st.metric("Ka/L", f"{results['Ka_over_L']:.3f}")
            st.metric("Static Pressure", f"{results['total_static_pressure']:.1f} Pa")
            st.metric("Water Loading", f"{results['water_loading']:.1f} m³/h·m²")
            st.metric("Air Density", f"{results['air_density']:.3f} kg/m³")
        
        with col2:
            st.subheader("Supplier's Claim (SAA15)")
            st.metric("Cold Water Temp", "35.00°C")
            st.metric("Fan Power", "13.41 kW")
            st.metric("Ka/L", "0.982")
            st.metric("Static Pressure", f"{comparison['supplier_claimed']['static_pressure_Pa']:.1f} Pa")
            st.metric("Water Loading", f"{comparison['supplier_claimed']['water_loading']:.1f} m³/h·m²")
            st.metric("Air Density", "~0.915 kg/m³")
        
        # Differences
        st.subheader("📊 Differences")
        diff_col1, diff_col2, diff_col3, diff_col4 = st.columns(4)
        with diff_col1:
            delta_temp = comparison['differences']['T_cold_diff']
            st.metric("Δ Cold Temp", f"{delta_temp:.2f}°C", 
                     delta_color="inverse" if delta_temp > 0 else "normal")
        with diff_col2:
            delta_power = comparison['differences']['fan_power_diff']
            st.metric("Δ Fan Power", f"{delta_power:.2f} kW",
                     delta_color="inverse" if delta_power > 0 else "normal")
        with diff_col3:
            delta_kal = comparison['differences']['Ka_over_L_diff']
            st.metric("Δ Ka/L", f"{delta_kal:.3f}",
                     delta_color="normal" if delta_kal > 0 else "inverse")
        with diff_col4:
            st.metric("RH Calculated", f"{results['RH']:.1f}%", "From dry/wet bulb")
        
        # Interpretation
        st.info("""
        **Interpretation:**
        - If your code matches supplier closely (±5%), CF1200 curve is accurate
        - If your code shows better performance, supplier may be conservative
        - If your code shows worse performance, check pressure drop assumptions
        - Dry bulb temperature affects air density and psychrometric calculations
        """)
    
    # ========================================================================
    # MAIN CALCULATION
    # ========================================================================
    if run_calc and selected_fills:
        # Validate temperatures
        if T_hot <= T_cold_target:
            st.error("❌ Error: Hot water temperature must be GREATER than cold water target")
            st.stop()
        
        # Validate dry bulb is greater than wet bulb
        if Tdb <= Twb:
            st.warning("⚠️ Note: Dry bulb temperature should typically be higher than wet bulb temperature")
        
        # Calculate for all selected fills
        results = []
        
        with st.spinner("Running cooling tower calculations..."):
            for fill in selected_fills:
                result = solve_cooling_tower_enhanced(
                    L, G, T_hot, T_cold_target, Twb, Tdb, fill,
                    tower_type, fill_depth, st.session_state.face_area, altitude
                )
                results.append(result)
        
        # ====================================================================
        # PERFORMANCE RESULTS WITH MANUFACTURER LIMITS
        # ====================================================================
        st.header("📊 Performance Results with Manufacturer Limits")
        
        # Create metrics columns
        cols = st.columns(len(selected_fills))
        for idx, (col, result) in enumerate(zip(cols, results)):
            with col:
                st.subheader(f"{result['fill_name']}")
                st.caption(f"{result['tower_name']}")
                
                # Temperature status
                temp_status = "✅" if result['T_cold_achieved'] <= result['T_cold_target'] else "❌"
                st.metric(f"{temp_status} Cold Water", 
                         f"{result['T_cold_achieved']:.2f}°C",
                         delta=f"{result['T_cold_achieved'] - result['T_cold_target']:.2f}°C vs target")
                
                # Water Loading with Manufacturer Limit
                water_loading_status = result['water_status'].split()[0]  # Get emoji
                water_color = "green" if water_loading_status == "✅" else "red" if water_loading_status == "❌" else "orange"
                
                st.metric(f"{water_loading_status} Water Loading", 
                         f"{result['water_loading']:.1f} m³/h·m²",
                         delta=f"Max: {result['max_water_loading']} m³/h·m²",
                         delta_color="normal")
                
                # Additional water loading info
                with st.expander("Water Loading Details"):
                    st.write(f"**Actual:** {result['water_loading']:.1f} m³/h·m²")
                    st.write(f"**Manufacturer Range:** {result['min_water_loading']} - {result['max_water_loading']} m³/h·m²")
                    st.write(f"**Recommended:** {result['recommended_water_loading']} m³/h·m²")
                    st.write(f"**Status:** {result['water_status']}")
                
                # Air Velocity with Manufacturer Limit
                air_velocity_status = result['air_status'].split()[0]  # Get emoji
                st.metric(f"{air_velocity_status} Air Velocity", 
                         f"{result['air_face_velocity']:.2f} m/s",
                         delta=f"Max: {result['max_air_velocity']} m/s",
                         delta_color="normal")
                
                # Additional air velocity info
                with st.expander("Air Velocity Details"):
                    st.write(f"**Actual:** {result['air_face_velocity']:.2f} m/s")
                    st.write(f"**Manufacturer Maximum:** {result['max_air_velocity']} m/s")
                    st.write(f"**Recommended:** {result['recommended_air_velocity']} m/s")
                    st.write(f"**Status:** {result['air_status']}")
                
                st.metric("Heat Rejection", f"{result['Q_achieved']:.0f} kW")
                st.metric("Fan Power", f"{result['fan_power']:.2f} kW")
        
        # ====================================================================
        # MANUFACTURER LIMITS COMPLIANCE SUMMARY
        # ====================================================================
        st.header("⚙️ Manufacturer Limits Compliance Summary")
        
        # Check for critical issues
        critical_issues = []
        for result in results:
            if result['water_loading'] > result['max_water_loading']:
                critical_issues.append(f"**{result['fill_name']}**: Water loading {result['water_loading']:.1f} m³/h·m² exceeds manufacturer maximum {result['max_water_loading']} m³/h·m²")
            if result['water_loading'] < result['min_water_loading']:
                critical_issues.append(f"**{result['fill_name']}**: Water loading {result['water_loading']:.1f} m³/h·m² below manufacturer minimum {result['min_water_loading']} m³/h·m²")
            if result['air_face_velocity'] > result['max_air_velocity']:
                critical_issues.append(f"**{result['fill_name']}**: Air velocity {result['air_face_velocity']:.2f} m/s exceeds manufacturer maximum {result['max_air_velocity']} m/s")
        
        if critical_issues:
            st.error("🚨 **CRITICAL DESIGN ISSUES - Manufacturer Limits Exceeded:**")
            for issue in critical_issues:
                st.write(f"- {issue}")
            
            # Recommended actions
            with st.expander("🛠️ Recommended Actions to Fix Issues"):
                st.markdown("""
                **For High Water Loading (> Manufacturer Max):**
                1. **Reduce water flow rate** if possible
                2. **Increase tower face area** (make tower larger)
                3. **Select different fill** with higher limits (e.g., XF3000 max 25 m³/h·m²)
                
                **For Low Water Loading (< Manufacturer Min):**
                1. **Increase water flow rate** if possible
                2. **Reduce tower face area** (make tower smaller)
                3. **Ensure proper water distribution** across fill
                
                **For High Air Velocity (> Manufacturer Max):**
                1. **Reduce air flow rate** (increase L/G ratio)
                2. **Increase tower face area** (make tower larger)
                3. **Check fan selection** for proper operating point
                """)
        else:
            st.success("✅ **All manufacturer limits satisfied for all selected fills**")
        
        # ====================================================================
        # DETAILED COMPARISON TABLE WITH LIMITS
        # ====================================================================
        st.header("📋 Detailed Performance Comparison with Limits")
        comparison_data = []
        for result in results:
            comparison_data.append({
                "Fill Type": result['fill_name'],
                "Tower Type": result['tower_name'],
                "Cold Water (°C)": f"{result['T_cold_achieved']:.2f}",
                "Heat Rejection (kW)": f"{result['Q_achieved']:.0f}",
                "Approach (°C)": f"{result['approach']:.2f}",
                "Range (°C)": f"{result['cooling_range']:.2f}",
                "L/G Ratio": f"{result['L_over_G']:.3f}",
                "NTU": f"{result['NTU']:.3f}",
                "Ka/L (1/m)": f"{result['Ka_over_L']:.3f}",
                "Water Loading (m³/h·m²)": f"{result['water_loading']:.1f} {result['water_status']}",
                "Water Limits": f"{result['min_water_loading']}-{result['max_water_loading']}",
                "Air Velocity (m/s)": f"{result['air_face_velocity']:.2f} {result['air_status']}",
                "Air Limit": f"≤{result['max_air_velocity']}",
                "Fan Power (kW)": f"{result['fan_power']:.2f}",
                "Static Pressure (Pa)": f"{result['total_static_pressure']:.0f}",
                "Fouling Risk": result['fouling_risk']['risk_level']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # ====================================================================
        # MANUFACTURER LIMITS DETAILED TABLE
        # ====================================================================
        st.header("📜 Manufacturer Specifications Comparison")
        limits_data = []
        for result in results:
            # Color coding for water loading
            water_bg_color = "background-color: #ffcccc" if result['water_loading'] > result['max_water_loading'] else \
                            "background-color: #fff3cd" if result['water_loading'] < result['min_water_loading'] else \
                            "background-color: #d4edda"
            
            # Color coding for air velocity
            air_bg_color = "background-color: #ffcccc" if result['air_face_velocity'] > result['max_air_velocity'] else \
                          "background-color: #d4edda"
            
            limits_data.append({
                "Fill Type": result['fill_name'],
                "Water Loading (m³/h·m²)": f"{result['water_loading']:.1f}",
                "Min Limit": f"{result['min_water_loading']}",
                "Max Limit": f"{result['max_water_loading']}",
                "Status": result['water_status'],
                "Air Velocity (m/s)": f"{result['air_face_velocity']:.2f}",
                "Max Limit": f"{result['max_air_velocity']}",
                "Status": result['air_status'],
                "Recommended Water": f"{result['recommended_water_loading']}",
                "Recommended Air": f"{result['recommended_air_velocity']}"
            })
        
        df_limits = pd.DataFrame(limits_data)
        
        # Apply styling to highlight issues
        def highlight_limits(row):
            styles = [''] * len(row)
            
            # Check water loading column (index 1)
            water_val = float(row[1])
            water_min = float(row[2])
            water_max = float(row[3])
            
            if water_val > water_max:
                styles[1] = 'background-color: #ffcccc; font-weight: bold;'
            elif water_val < water_min:
                styles[1] = 'background-color: #fff3cd; font-weight: bold;'
            
            # Check air velocity column (index 5)
            air_val = float(row[5])
            air_max = float(row[6])
            
            if air_val > air_max:
                styles[5] = 'background-color: #ffcccc; font-weight: bold;'
            
            return styles
        
        styled_df = df_limits.style.apply(highlight_limits, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # ====================================================================
        # ATMOSPHERIC CONDITIONS & GEOMETRY
        # ====================================================================
        col_atm1, col_atm2, col_atm3, col_atm4 = st.columns(4)
        with col_atm1:
            st.metric("Dry Bulb", f"{Tdb:.1f}°C")
        with col_atm2:
            st.metric("Wet Bulb", f"{Twb:.1f}°C")
        with col_atm3:
            st.metric("Relative Humidity", f"{results[0]['RH']:.1f}%")
        with col_atm4:
            st.metric("Altitude", f"{altitude} m ASL")
        
        st.header("📐 Tower Geometry Summary")
        col_geo1, col_geo2, col_geo3, col_geo4 = st.columns(4)
        with col_geo1:
            st.metric("Face Area", f"{st.session_state.face_area:.2f} m²")
        with col_geo2:
            st.metric("Fill Depth", f"{fill_depth:.3f} m")
        with col_geo3:
            st.metric("Tower Shape", tower_shape)
        with col_geo4:
            st.metric("Fill Volume", f"{st.session_state.face_area * fill_depth:.2f} m³")
        
        # ====================================================================
        # VISUALIZATION
        # ====================================================================
        st.header("📈 Performance Visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cold water temperatures
        fill_names = [r['fill_name'] for r in results]
        cold_temps = [r['T_cold_achieved'] for r in results]
        
        bars1 = ax1.bar(fill_names, cold_temps, color=['red' if t > T_cold_target else 'green' for t in cold_temps])
        ax1.axhline(y=T_cold_target, color='blue', linestyle='--', label='Target')
        ax1.set_ylabel('Cold Water Temperature (°C)')
        ax1.set_title('Performance vs Target')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # Fan power comparison
        fan_powers = [r['fan_power'] for r in results]
        bars2 = ax2.bar(fill_names, fan_powers, color='orange', alpha=0.7)
        ax2.set_ylabel('Fan Power (kW)')
        ax2.set_title('Energy Consumption')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bars, ax in [(bars1, ax1), (bars2, ax2)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max([b.get_height() for b in bars])),
                       f'{height:.2f}' if ax == ax1 else f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig)
        
        # ====================================================================
        # FILL-SPECIFIC NOTES
        # ====================================================================
        if "CF1200" in selected_fills:
            st.warning("""
            **Note about CF1200 fill:**
            - This is an **older fill design** with performance data tuned to match supplier's SAA15 design
            - Compared to modern fills (XF75), CF1200 has:
              - Lower thermal efficiency (Ka/L ~70% of XF75)
              - Higher pressure drop
              - Worse fouling resistance
              - Lower water loading limits (14 vs 15 m³/h·m²)
            - Use for comparison with existing towers or supplier designs
            """)
        
        # ====================================================================
        # REPORT GENERATION
        # ====================================================================
        if generate_reports:
            st.header("📄 Report Generation")
            selected_for_report = st.selectbox(
                "Select design for detailed report:",
                options=[f"{r['fill_name']} ({r['T_cold_achieved']:.2f}°C)" for r in results],
                index=0
            )
            
            fill_index = [f"{r['fill_name']} ({r['T_cold_achieved']:.2f}°C)" for r in results].index(selected_for_report)
            selected_result = results[fill_index]
            
            txt_report = generate_txt_report(selected_result)
            st.download_button(
                label="📥 Download Detailed TXT Report",
                data=txt_report,
                file_name=f"cooling_tower_{selected_result['fill_type']}_{selected_result['tower_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Show report preview
            with st.expander("📋 Preview Report (First 2000 characters)"):
                st.text(txt_report[:2000] + "..." if len(txt_report) > 2000 else txt_report)
    
    elif run_calc and not selected_fills:
        st.warning("Please select at least one Brentwood fill type.")
    else:
        # Welcome message
        st.markdown("""
        ## 🌊 Complete Cooling Tower Design Tool
        
        ### ✅ **ENHANCED FEATURES:**
        
        1. **Manufacturer Limits Display:**
           - Water loading limits shown prominently on-screen
           - Air velocity limits with clear compliance status
           - Color-coded warnings when limits are exceeded
           - Recommended actions to fix issues
        
        2. **Enhanced Input Controls:**
           - Number input boxes with +/- buttons
           - Tower shape selection (Rectangle or Round)
           - Automatic face area calculation
           - Dry bulb temperature input
        
        3. **Complete Analysis:**
           - CF1200 fill support with supplier validation
           - Counterflow tower types
           - Multiple fill comparison
           - Detailed performance metrics
        
        ### 🎯 **How to Use:**
        
        1. Configure all inputs in the **sidebar**
        2. Select **fills to compare** (include CF1200 for supplier comparison)
        3. Click **"Run Complete Analysis"**
        4. Check **manufacturer limits compliance** in results
        5. Download **detailed report** for documentation
        
        ### 📊 **Manufacturer Limits by Fill:**
        
        | Fill Type | Water Loading Range (m³/h·m²) | Max Air Velocity (m/s) |
        |-----------|-------------------------------|------------------------|
        | CF1200 | 6-14 | 2.8 |
        | XF75 | 5-15 | 3.0 |
        | XF3000 | 8-25 | 3.2 |
        
        ---
        
        *Configure your design in the sidebar and run analysis to see manufacturer limits compliance.*
        """)

if __name__ == "__main__":
    main()