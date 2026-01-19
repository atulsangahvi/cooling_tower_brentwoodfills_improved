# cooling_tower_design_complete.py
# Complete Cooling Tower Design Tool with ALL Features + Geometric Enhancements

import streamlit as st
import numpy as np
import pandas as pd

# Try to import matplotlib with error handling
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    st.error("⚠️ matplotlib is not installed. Please add it to requirements.txt")
    st.stop()

# Other imports
from io import BytesIO
import datetime
import hashlib

# ============================================================================
# PASSWORD PROTECTION
# ============================================================================

def check_password():
    """Returns `True` if the user entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Hash the input password
        input_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()
        correct_hash = hashlib.sha256("Semaanju".encode()).hexdigest()
        
        if input_hash == correct_hash:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False
    
    # First run, show input for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.markdown("*Hint: The password is 'Semaanju'*")
        return False
    # Password incorrect
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    # Password correct
    else:
        return True

# ============================================================================
# ENHANCED BRENTWOOD FILL DATABASE
# ============================================================================

BRENTWOOD_FILLS = {
    "XF75": {
        "name": "Brentwood XF75",
        "surface_area": 226,  # m²/m³
        "sheet_spacing": 11.7,  # mm
        "flute_angle": 30,  # degrees
        "channel_depth": 9.0,  # mm (estimated)
        "channel_width": 13.5,  # mm (estimated)
        "hydraulic_diameter": 8.8,  # mm
        "free_area_fraction": 0.89,
        "water_passage_area": 0.78,
        "material_thickness_options": [0.20, 0.25, 0.30],
        "dry_weight_range": [36.8, 60.9],
        "water_film_thickness": 0.6,  # mm
        "max_water_loading": 15,  # m³/h·m²
        "min_water_loading": 5,
        "recommended_air_velocity": 2.5,  # m/s
        "max_air_velocity": 3.0,
        "fouling_factor": 0.85,
        
        # Performance curves
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
# PSYCHROMETRIC FUNCTIONS
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
    
    h_fg = 2501.0  # kJ/kg at 0°C
    Cp_air = 1.006  # kJ/kg°C
    Cp_vapor = 1.86  # kJ/kg°C
    
    W = ((h_fg - Cp_vapor * wb) * Ws_wb - Cp_air * (db - wb)) / (h_fg + Cp_vapor * db - 4.186 * wb)
    return max(W, 0.0001)

def enthalpy_air(db, W):
    """Calculate enthalpy of moist air in kJ/kg dry air"""
    Cp_air = 1.006  # kJ/kg°C
    Cp_vapor = 1.86  # kJ/kg°C
    h_fg = 2501.0  # kJ/kg at 0°C
    return Cp_air * db + W * (h_fg + Cp_vapor * db)

# ============================================================================
# ENHANCED CALCULATION FUNCTIONS
# ============================================================================

def calculate_hydraulic_properties(fill_data, water_loading, air_face_velocity):
    """Calculate hydraulic properties based on fill geometry"""
    # Convert water loading from m³/h·m² to m/s in channels
    water_velocity_ms = (water_loading / 3.6) / fill_data["water_passage_area"]
    
    # Water film Reynolds number
    water_viscosity = 1e-6  # m²/s at 30°C
    film_reynolds = (water_velocity_ms * fill_data["water_film_thickness"] * 1e-3) / water_viscosity
    
    # Air side Reynolds number
    air_viscosity = 1.5e-5  # m²/s at 30°C
    air_reynolds = (air_face_velocity * fill_data["hydraulic_diameter"] * 1e-3) / air_viscosity
    
    return {
        "water_velocity": water_velocity_ms,  # m/s
        "film_reynolds": film_reynolds,
        "air_reynolds": air_reynolds,
        "water_film_thickness": fill_data["water_film_thickness"]  # mm
    }

def assess_fouling_risk(fill_data, hydraulic_props):
    """Assess fouling risk based on fill geometry"""
    risk_score = 0
    
    # Small hydraulic diameter increases risk
    if fill_data["hydraulic_diameter"] < 10:
        risk_score += 2
    elif fill_data["hydraulic_diameter"] < 12:
        risk_score += 1
    
    # Low water velocity increases risk
    if hydraulic_props["water_velocity"] < 0.05:  # m/s
        risk_score += 2
    elif hydraulic_props["water_velocity"] < 0.1:
        risk_score += 1
    
    # Risk categories
    if risk_score < 2:
        risk_level = "Low"
    elif risk_score < 4:
        risk_level = "Moderate"
    elif risk_score < 6:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level
    }

# ============================================================================
# MERKEL EQUATION SOLVER
# ============================================================================

def solve_cooling_tower(L, G, T_hot, T_cold_target, Twb, fill_type, fill_depth, face_area):
    """
    Solve cooling tower using Merkel equation with geometric enhancements
    """
    fill_data = BRENTWOOD_FILLS[fill_type]
    
    # Get Ka/L from performance curve
    L_over_G = L / G
    Ka_over_L = np.interp(L_over_G, 
                         fill_data["performance_data"]["L_G"], 
                         fill_data["performance_data"]["Ka_L"])
    
    # Total heat transfer coefficient
    Ka = Ka_over_L * L  # kW/°C
    
    # Air properties
    air_density = 1.2  # kg/m³
    air_flow_volumetric = G / air_density  # m³/s
    air_face_velocity = air_flow_volumetric / face_area  # m/s
    
    # Water loading
    water_loading = (L * 3.6) / face_area  # m³/h·m²
    
    # Calculate hydraulic properties
    hydraulic_props = calculate_hydraulic_properties(fill_data, water_loading, air_face_velocity)
    
    # Assess fouling risk
    fouling_risk = assess_fouling_risk(fill_data, hydraulic_props)
    
    # Pressure drop calculation
    delta_P_base = np.interp(L_over_G, 
                            fill_data["performance_data"]["L_G"], 
                            fill_data["performance_data"]["delta_P_base"])
    
    # Adjust for actual face velocity (ΔP ∝ velocity²)
    velocity_factor = (air_face_velocity / 2.5) ** 2
    fill_pressure_drop = delta_P_base * velocity_factor * fill_depth
    total_static_pressure = fill_pressure_drop * 1.35  # Include other losses
    
    # Merkel number (NTU)
    Cp = 4.186  # kJ/kg°C
    NTU = Ka_over_L * fill_depth
    
    # Achieved cold water temperature (simplified Merkel solution)
    T_cold_achieved = Twb + (T_hot - Twb) * np.exp(-NTU)
    Q_achieved = L * Cp * (T_hot - T_cold_achieved)
    
    # Fill volume and surface area
    fill_volume = face_area * fill_depth
    total_surface_area = fill_volume * fill_data["surface_area"]
    
    # Operating warnings
    operating_warnings = []
    if water_loading > fill_data["max_water_loading"]:
        operating_warnings.append(f"Water loading exceeds maximum ({fill_data['max_water_loading']} m³/h·m²)")
    if water_loading < fill_data["min_water_loading"]:
        operating_warnings.append(f"Water loading below minimum ({fill_data['min_water_loading']} m³/h·m²)")
    if air_face_velocity > fill_data["max_air_velocity"]:
        operating_warnings.append(f"Air face velocity exceeds maximum ({fill_data['max_air_velocity']} m/s)")
    
    return {
        # Basic identification
        "fill_type": fill_type,
        "fill_name": fill_data["name"],
        
        # Temperatures and heat transfer
        "T_hot": T_hot,
        "T_cold_achieved": T_cold_achieved,
        "T_cold_target": T_cold_target,
        "Twb": Twb,
        "Q_achieved": Q_achieved,
        "Q_target": L * Cp * (T_hot - T_cold_target),
        "approach": T_cold_achieved - Twb,
        "cooling_range": T_hot - T_cold_achieved,
        
        # Flow parameters
        "L": L,
        "G": G,
        "L_over_G": L_over_G,
        "water_loading": water_loading,
        "air_flow_volumetric": air_flow_volumetric,
        "air_face_velocity": air_face_velocity,
        
        # Geometry and sizing
        "fill_depth": fill_depth,
        "face_area": face_area,
        "fill_volume": fill_volume,
        "total_surface_area": total_surface_area,
        
        # Hydraulic properties
        "water_velocity": hydraulic_props["water_velocity"],
        "film_reynolds": hydraulic_props["film_reynolds"],
        "air_reynolds": hydraulic_props["air_reynolds"],
        "water_film_thickness": hydraulic_props["water_film_thickness"],
        
        # Performance parameters
        "NTU": NTU,
        "Ka_over_L": Ka_over_L,
        "Ka": Ka,
        
        # Pressure drop
        "fill_pressure_drop": fill_pressure_drop,
        "total_static_pressure": total_static_pressure,
        
        # Assessments
        "fouling_risk": fouling_risk,
        "operating_warnings": operating_warnings,
        
        # Fill characteristics
        "surface_area_density": fill_data["surface_area"],
        "hydraulic_diameter": fill_data["hydraulic_diameter"],
        "flute_angle": fill_data["flute_angle"],
        "free_area_fraction": fill_data["free_area_fraction"]
    }

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_txt_report(design_results):
    """Generate a detailed TXT report"""
    report = []
    report.append("=" * 70)
    report.append("COOLING TOWER DESIGN REPORT")
    report.append("=" * 70)
    report.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Fill: {design_results['fill_name']}")
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
    report.append(f"Fill Depth: {design_results['fill_depth']:.2f} m")
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
    report.append(f"Fan Airflow: {design_results['air_flow_volumetric']:.2f} m³/s")
    report.append(f"Fan Static Pressure: {design_results['total_static_pressure']:.1f} Pa")
    
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
        report.append("✅ DESIGN MEETS REQUIREMENTS")
    else:
        report.append("⚠️ DESIGN DOES NOT MEET REQUIREMENTS")
        report.append(f"   Required improvement: {design_results['T_cold_achieved'] - design_results['T_cold_target']:.2f} °C")
    
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
# STREAMLIT APP - MAIN FUNCTION
# ============================================================================

def main():
    # Check password first
    if not check_password():
        st.stop()
    
    # Set page config
    st.set_page_config(
        page_title="Professional Cooling Tower Design",
        page_icon="🌊",
        layout="wide"
    )
    
    # Main title
    st.title("🌊 Complete Cooling Tower Design Tool")
    st.markdown("**Two Calculation Modes | Brentwood Fills | Geometric Analysis**")
    
    # ========================================================================
    # SIDEBAR - DESIGN INPUTS (RESTORED ALL OPTIONS)
    # ========================================================================
    with st.sidebar:
        st.header("📥 Design Inputs")
        
        # ====================================================================
        # RESTORED: Two Calculation Modes
        # ====================================================================
        calc_mode = st.radio(
            "**Calculation Mode:**",
            ["Mode 1: Given Heat Load → Find Water Flow",
             "Mode 2: Given Water Flow → Find Heat Load"],
            help="Choose whether you know the heat load or water flow rate"
        )
        
        # Common temperature inputs
        col1, col2 = st.columns(2)
        with col1:
            T_hot = st.number_input("Hot Water In (°C)", 
                                   value=37.0, min_value=20.0, max_value=60.0, step=0.5)
        with col2:
            T_cold_target = st.number_input("Target Cold Water Out (°C)", 
                                           value=32.0, min_value=10.0, max_value=40.0, step=0.5)
        
        Twb = st.number_input("Ambient Wet Bulb (°C)", 
                             value=28.0, min_value=10.0, max_value=40.0, step=0.5)
        
        # ====================================================================
        # Mode-specific inputs (RESTORED)
        # ====================================================================
        if calc_mode == "Mode 1: Given Heat Load → Find Water Flow":
            Q_input = st.number_input("Heat Load to Remove (kW)", 
                                     value=2090.0, min_value=100.0, step=100.0)
            # Calculate water flow from heat load
            Cp = 4.186
            if T_hot > T_cold_target:
                L = Q_input / (Cp * (T_hot - T_cold_target))
            else:
                L = 100.0
                st.error("Hot water temperature must be greater than cold water target")
            st.metric("Calculated Water Flow", f"{L:.2f} kg/s")
        else:  # Mode 2
            L = st.number_input("Water Flow Rate (kg/s)", 
                               value=100.0, min_value=10.0, step=5.0)
            # Calculate heat load from water flow
            Cp = 4.186
            if T_hot > T_cold_target:
                Q_input = L * Cp * (T_hot - T_cold_target)
            else:
                Q_input = 2090.0
                st.error("Hot water temperature must be greater than cold water target")
            st.metric("Calculated Heat Load", f"{Q_input:.0f} kW")
        
        # ====================================================================
        # RESTORED: L/G Ratio OR Direct Air Flow Input
        # ====================================================================
        st.header("🌬️ Air Flow Specification")
        
        air_input_method = st.radio(
            "Air Flow Input Method:",
            ["Method 1: Set L/G Ratio",
             "Method 2: Set Direct Air Flow Rate"],
            help="Choose to set L/G ratio or direct air flow rate"
        )
        
        if air_input_method == "Method 1: Set L/G Ratio":
            L_over_G = st.slider(
                "L/G Ratio (Liquid to Gas mass ratio)", 
                min_value=0.5, max_value=2.0, value=1.25, step=0.05,
                help="Typical range: 0.8-1.5. Higher = more air, lower pressure drop"
            )
            G = L / L_over_G  # Calculate air flow from L/G ratio
            st.metric("Calculated Air Flow", f"{G:.2f} kg/s")
        else:  # Method 2: Direct Air Flow
            G = st.number_input("Air Mass Flow Rate (kg/s)", 
                               value=80.0, min_value=10.0, step=5.0)
            L_over_G = L / G
            st.metric("Calculated L/G Ratio", f"{L_over_G:.3f}")
        
        # ====================================================================
        # Geometry Parameters
        # ====================================================================
        st.header("📐 Geometry Parameters")
        
        fill_depth = st.slider(
            "Fill Depth (m)", 
            min_value=0.3, max_value=2.0, value=0.6, step=0.1,
            help="Depth of fill media in air flow direction"
        )
        
        face_area = st.slider(
            "Face Area (m²)", 
            min_value=10.0, max_value=100.0, value=36.94, step=5.0,
            help="Cross-sectional area perpendicular to air flow"
        )
        
        # ====================================================================
        # Brentwood Fill Selection
        # ====================================================================
        st.header("🎯 Brentwood Fill Selection")
        fill_options = list(BRENTWOOD_FILLS.keys())
        selected_fills = st.multiselect(
            "Select fills to compare:",
            options=fill_options,
            default=["XF75", "XF125", "XF3000"],
            format_func=lambda x: BRENTWOOD_FILLS[x]["name"]
        )
        
        # ====================================================================
        # Run Button
        # ====================================================================
        run_calc = st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)
        
        # Report generation
        st.header("📄 Report Generation")
        generate_reports = st.checkbox("Generate TXT Report", value=True)
    
    # ========================================================================
    # MAIN CONTENT - RESULTS
    # ========================================================================
    if run_calc and selected_fills:
        # Validate temperatures
        if T_hot <= T_cold_target:
            st.error("❌ Error: Hot water temperature must be GREATER than cold water target")
            st.stop()
        
        # Calculate for all selected fills
        results = []
        
        with st.spinner("Running cooling tower calculations..."):
            for fill in selected_fills:
                result = solve_cooling_tower(
                    L, G, T_hot, T_cold_target, Twb, fill,
                    fill_depth, face_area
                )
                results.append(result)
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        st.header("📊 Performance Results")
        
        # Create metrics columns for each fill
        cols = st.columns(len(selected_fills))
        for idx, (col, result) in enumerate(zip(cols, results)):
            with col:
                st.subheader(result['fill_name'])
                
                # Temperature status
                temp_status = "✅" if result['T_cold_achieved'] <= result['T_cold_target'] else "❌"
                st.metric(f"{temp_status} Cold Water Achieved", 
                         f"{result['T_cold_achieved']:.2f}°C",
                         delta=f"{result['T_cold_achieved'] - result['T_cold_target']:.2f}°C vs target")
                
                st.metric("Heat Rejection", f"{result['Q_achieved']:.0f} kW")
                st.metric("Fan Airflow", f"{result['air_flow_volumetric']:.2f} m³/s")
                st.metric("Static Pressure", f"{result['total_static_pressure']:.0f} Pa")
                st.metric("Water Loading", f"{result['water_loading']:.1f} m³/h·m²")
        
        # ====================================================================
        # DETAILED RESULTS TABLE (WITH ALL REQUESTED PARAMETERS)
        # ====================================================================
        st.header("📋 Detailed Performance Comparison")
        
        # Create detailed comparison table
        comparison_data = []
        for result in results:
            comparison_data.append({
                "Fill Type": result['fill_name'],
                "Cold Water (°C)": f"{result['T_cold_achieved']:.2f}",
                "Heat Rejection (kW)": f"{result['Q_achieved']:.0f}",
                "Approach (°C)": f"{result['approach']:.2f}",
                "Range (°C)": f"{result['cooling_range']:.2f}",
                "L/G Ratio": f"{result['L_over_G']:.3f}",
                "NTU": f"{result['NTU']:.3f}",
                "Ka/L (1/m)": f"{result['Ka_over_L']:.3f}",
                # NEW: Total Surface Area
                "Surface Area (m²)": f"{result['total_surface_area']:.0f}",
                # NEW: Water Velocity in Channels
                "Water Velocity (m/s)": f"{result['water_velocity']:.3f}",
                # NEW: Water Film Thickness
                "Film Thickness (mm)": f"{result['water_film_thickness']}",
                "Water Loading (m³/h·m²)": f"{result['water_loading']:.1f}",
                "Air Velocity (m/s)": f"{result['air_face_velocity']:.2f}",
                "Fan Airflow (m³/s)": f"{result['air_flow_volumetric']:.2f}",
                "Static Pressure (Pa)": f"{result['total_static_pressure']:.0f}",
                "Fouling Risk": result['fouling_risk']['risk_level']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # ====================================================================
        # GEOMETRIC ANALYSIS SECTION
        # ====================================================================
        st.header("📐 Geometric & Hydraulic Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Surface Areas", "Water Flow", "Air Flow"])
        
        with tab1:
            # Surface area comparison
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            fill_names = [r['fill_name'] for r in results]
            surface_areas = [r['total_surface_area'] for r in results]
            surface_densities = [r['surface_area_density'] for r in results]
            
            x = np.arange(len(fill_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, surface_areas, width, label='Total Surface Area', color='skyblue')
            ax1.set_xlabel('Fill Type')
            ax1.set_ylabel('Total Surface Area (m²)', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            ax1.set_xticks(x)
            ax1.set_xticklabels(fill_names, rotation=45)
            
            ax2 = ax1.twinx()
            bars2 = ax2.bar(x + width/2, surface_densities, width, label='Surface Area Density', color='salmon')
            ax2.set_ylabel('Surface Area Density (m²/m³)', color='salmon')
            ax2.tick_params(axis='y', labelcolor='salmon')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax = ax1 if bars == bars1 else ax2
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max([b.get_height() for b in bars])),
                           f'{height:.0f}' if bars == bars1 else f'{height:.0f}',
                           ha='center', va='bottom', fontsize=9)
            
            fig1.tight_layout()
            st.pyplot(fig1)
            
            # Explanation
            st.info("""
            **Surface Area Analysis:**
            - **Total Surface Area**: Actual heat transfer area in your design
            - **Surface Area Density**: How much area per cubic meter of fill
            - Higher density fills (XF75) provide more area in less volume
            """)
        
        with tab2:
            # Water flow analysis
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Water velocity comparison
            water_velocities = [r['water_velocity'] for r in results]
            bars1 = ax1.bar(fill_names, water_velocities, color='blue', alpha=0.7)
            ax1.set_ylabel('Water Velocity in Channels (m/s)')
            ax1.set_title('Water Flow Velocity')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add velocity guidelines
            ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Minimum (0.05 m/s)')
            ax1.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='Good (0.15 m/s)')
            ax1.legend(fontsize=8)
            
            # Water film thickness
            film_thickness = [r['water_film_thickness'] for r in results]
            bars2 = ax2.bar(fill_names, film_thickness, color='lightblue', alpha=0.7)
            ax2.set_ylabel('Water Film Thickness (mm)')
            ax2.set_title('Water Film Characteristics')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bars, ax in [(bars1, ax1), (bars2, ax2)]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max([b.get_height() for b in bars])),
                           f'{height:.3f}' if ax == ax1 else f'{height}',
                           ha='center', va='bottom', fontsize=9)
            
            fig2.tight_layout()
            st.pyplot(fig2)
            
            # Water flow explanation
            st.info("""
            **Water Flow Analysis:**
            - **Water Velocity**: Speed of water in fill channels
              - < 0.05 m/s: Risk of sedimentation and fouling
              - 0.1-0.2 m/s: Good operating range
              - > 0.3 m/s: Risk of excessive pressure drop
            - **Film Thickness**: Thickness of water film on fill surfaces
              - Thinner films: Better heat transfer
              - Thicker films: Better fouling resistance
            """)
        
        with tab3:
            # Air flow analysis
            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Air face velocity
            air_velocities = [r['air_face_velocity'] for r in results]
            bars1 = ax1.bar(fill_names, air_velocities, color='green', alpha=0.7)
            ax1.set_ylabel('Air Face Velocity (m/s)')
            ax1.set_title('Air Flow Velocity')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add velocity limits from fill data
            for idx, fill in enumerate(selected_fills):
                fill_data = BRENTWOOD_FILLS[fill]
                ax1.axhline(y=fill_data['max_air_velocity'], xmin=idx/len(fill_names), 
                           xmax=(idx+1)/len(fill_names), color='red', linestyle=':', alpha=0.5)
            
            # Static pressure comparison
            static_pressures = [r['total_static_pressure'] for r in results]
            bars2 = ax2.bar(fill_names, static_pressures, color='orange', alpha=0.7)
            ax2.set_ylabel('Static Pressure (Pa)')
            ax2.set_title('Fan Static Pressure Requirement')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bars, ax in [(bars1, ax1), (bars2, ax2)]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max([b.get_height() for b in bars])),
                           f'{height:.2f}' if ax == ax1 else f'{height:.0f}',
                           ha='center', va='bottom', fontsize=9)
            
            fig3.tight_layout()
            st.pyplot(fig3)
            
            # Air flow explanation
            st.info("""
            **Air Flow Analysis:**
            - **Air Face Velocity**: Speed of air through fill
              - Dotted lines show maximum recommended for each fill
              - Typical range: 2.0-3.0 m/s for crossflow
              - Higher velocity = better heat transfer but more pressure drop
            - **Static Pressure**: Fan power requirement
              - Higher for dense fills (XF75)
              - Lower for open fills (XF3000)
            """)
        
        # ====================================================================
        # FILL CHARACTERISTICS COMPARISON
        # ====================================================================
        st.header("⚙️ Fill Characteristics Comparison")
        
        # Create characteristic comparison
        characteristics_data = []
        for result in results:
            fill_data = BRENTWOOD_FILLS[result['fill_type']]
            characteristics_data.append({
                "Fill Type": result['fill_name'],
                "Surface Area (m²/m³)": fill_data['surface_area'],
                "Hydraulic Diameter (mm)": fill_data['hydraulic_diameter'],
                "Flute Angle (°)": fill_data['flute_angle'],
                "Sheet Spacing (mm)": fill_data['sheet_spacing'],
                "Free Area Fraction": f"{fill_data['free_area_fraction']:.2f}",
                "Max Water Loading (m³/h·m²)": fill_data['max_water_loading'],
                "Max Air Velocity (m/s)": fill_data['max_air_velocity'],
                "Fouling Factor": f"{fill_data['fouling_factor']:.2f}"
            })
        
        df_characteristics = pd.DataFrame(characteristics_data)
        st.dataframe(df_characteristics, use_container_width=True)
        
        # ====================================================================
        # DESIGN RECOMMENDATIONS
        # ====================================================================
        st.header("🎯 Design Recommendations")
        
        if len(results) > 1:
            # Find best fill for different criteria
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Best for cold water temperature
                best_temp = min(results, key=lambda x: x['T_cold_achieved'])
                st.metric("Best Cooling", best_temp['fill_name'], f"{best_temp['T_cold_achieved']:.2f}°C")
            
            with col2:
                # Lowest pressure drop
                best_pressure = min(results, key=lambda x: x['total_static_pressure'])
                st.metric("Lowest ΔP", best_pressure['fill_name'], f"{best_pressure['total_static_pressure']:.0f} Pa")
            
            with col3:
                # Highest surface area
                best_surface = max(results, key=lambda x: x['total_surface_area'])
                st.metric("Most Area", best_surface['fill_name'], f"{best_surface['total_surface_area']:.0f} m²")
            
            with col4:
                # Best fouling resistance
                best_fouling = min(results, key=lambda x: x['fouling_risk']['risk_score'])
                st.metric("Lowest Fouling Risk", best_fouling['fill_name'], best_fouling['fouling_risk']['risk_level'])
        
        # ====================================================================
        # REPORT GENERATION
        # ====================================================================
        if generate_reports:
            st.header("📄 Report Generation")
            
            # Let user select which fill to generate report for
            selected_for_report = st.selectbox(
                "Select fill for detailed report:",
                options=[f"{r['fill_name']} ({r['T_cold_achieved']:.2f}°C)" 
                        for r in results],
                index=0
            )
            
            # Extract the fill from selection
            fill_index = [f"{r['fill_name']} ({r['T_cold_achieved']:.2f}°C)" 
                         for r in results].index(selected_for_report)
            selected_result = results[fill_index]
            
            # Generate and download TXT report
            txt_report = generate_txt_report(selected_result)
            st.download_button(
                label="📥 Download Detailed TXT Report",
                data=txt_report,
                file_name=f"cooling_tower_{selected_result['fill_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Show report preview
            with st.expander("📋 Preview Report"):
                st.text(txt_report[:1500] + "..." if len(txt_report) > 1500 else txt_report)
        
        # ====================================================================
        # DETAILED RESULTS FOR EACH FILL
        # ====================================================================
        st.header("🔬 Detailed Results for Each Fill")
        
        for result in results:
            with st.expander(f"{result['fill_name']} - Complete Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Performance Metrics:**")
                    st.write(f"- **Cold Water Achieved**: {result['T_cold_achieved']:.2f}°C")
                    st.write(f"- **Heat Rejection**: {result['Q_achieved']:.0f} kW")
                    st.write(f"- **Approach**: {result['approach']:.2f}°C")
                    st.write(f"- **Range**: {result['cooling_range']:.2f}°C")
                    st.write(f"- **NTU**: {result['NTU']:.3f}")
                    st.write(f"- **Ka/L**: {result['Ka_over_L']:.3f} 1/m")
                    st.write(f"- **L/G Ratio**: {result['L_over_G']:.3f}")
                    
                    # Status
                    if result["T_cold_achieved"] <= result["T_cold_target"]:
                        st.success("✅ **Target Achieved**")
                    else:
                        st.error(f"❌ **Target NOT Achieved** (by {result['T_cold_achieved'] - result['T_cold_target']:.2f}°C)")
                
                with col2:
                    st.markdown("**Geometry & Hydraulics:**")
                    st.write(f"- **Total Surface Area**: {result['total_surface_area']:.0f} m²")
                    st.write(f"- **Water Loading**: {result['water_loading']:.1f} m³/h·m²")
                    st.write(f"- **Water Velocity**: {result['water_velocity']:.3f} m/s")
                    st.write(f"- **Water Film Thickness**: {result['water_film_thickness']} mm")
                    st.write(f"- **Air Face Velocity**: {result['air_face_velocity']:.2f} m/s")
                    st.write(f"- **Fan Airflow**: {result['air_flow_volumetric']:.2f} m³/s")
                    st.write(f"- **Static Pressure**: {result['total_static_pressure']:.0f} Pa")
                    st.write(f"- **Fouling Risk**: {result['fouling_risk']['risk_level']}")
                
                # Show warnings if any
                if result['operating_warnings']:
                    st.warning("**Operating Warnings:**")
                    for warning in result['operating_warnings']:
                        st.write(f"- {warning}")
    
    elif run_calc and not selected_fills:
        st.warning("Please select at least one Brentwood fill type.")
    else:
        # Welcome message
        st.markdown("""
        ## 🌊 Complete Cooling Tower Design Tool
        
        ### ✅ **RESTORED Features:**
        
        1. **Two Calculation Modes:**
           - **Mode 1**: Given Heat Load → Find Water Flow
           - **Mode 2**: Given Water Flow → Find Heat Load
        
        2. **Flexible Air Flow Input:**
           - **Method 1**: Set L/G Ratio (Liquid to Gas mass ratio)
           - **Method 2**: Set Direct Air Flow Rate
        
        3. **Enhanced Outputs:**
           - Total Surface Area (m²)
           - Water Velocity in Channels (m/s)
           - Water Film Thickness (mm)
           - All geometric parameters
        
        ### 🎯 **How to Use:**
        
        1. **Select calculation mode** (Heat Load or Water Flow)
        2. **Input temperatures** (Hot water, Target cold, Wet bulb)
        3. **Choose air flow method** (L/G ratio or direct air flow)
        4. **Set geometry** (Fill depth, Face area)
        5. **Select Brentwood fills** to compare
        6. **Click "Run Complete Analysis"**
        
        ### 📊 **New Outputs Included:**
        
        - **Total Surface Area**: Actual heat transfer area in your design
        - **Water Velocity**: Speed of water in fill channels
        - **Water Film Thickness**: Thickness of water film on fill surfaces
        - **Hydraulic Diameter**: Characteristic size of water channels
        - **Fouling Risk Assessment**: Based on geometry and operating conditions
        
        ---
        
        *Configure your design in the sidebar and click "Run Complete Analysis" to begin.*
        """)

if __name__ == "__main__":
    main()