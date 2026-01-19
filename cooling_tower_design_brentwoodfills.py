# cooling_tower_design_enhanced.py
# Enhanced with detailed geometric parameters for Brentwood fills

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
import base64

# ============================================================================
# ENHANCED BRENTWOOD FILL DATABASE WITH GEOMETRIC PARAMETERS
# ============================================================================

BRENTWOOD_FILLS_ENHANCED = {
    "XF75": {
        "name": "Brentwood XF75",
        "surface_area": 226,  # m²/m³
        "sheet_spacing": 11.7,  # mm
        "flute_angle": 30,  # degrees
        "channel_depth": 9.0,  # mm (estimated)
        "channel_width": 13.5,  # mm (estimated)
        "hydraulic_diameter": 8.8,  # mm (4*A/P)
        "free_area_fraction": 0.89,  # Air flow area / total area
        "water_passage_area": 0.78,  # Water flow area fraction
        "material_thickness_options": [0.20, 0.25, 0.30],  # mm
        "dry_weight_range": [36.8, 60.9],  # kg/m³
        "water_film_thickness": 0.6,  # mm (typical operating)
        "max_water_loading": 15,  # m³/h·m²
        "min_water_loading": 5,  # m³/h·m²
        "recommended_air_velocity": 2.5,  # m/s
        "max_air_velocity": 3.0,  # m/s
        "fouling_factor": 0.85,  # 1=clean, lower=more prone to fouling
        
        # Performance curves (Ka/L in 1/m)
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.3, 1.95, 1.65, 1.4, 1.2, 1.05, 0.92],
            "delta_P_base": [35, 45, 60, 80, 105, 135, 170]  # Pa/m at 2.5 m/s
        },
        
        "description": "High density cross-fluted fill with maximum surface area. Tight sheet spacing provides excellent heat transfer but requires good water quality.",
        "applications": ["High efficiency cooling", "Space-constrained installations", "Clean water systems"],
        "limitations": ["Sensitive to fouling", "Higher pressure drop", "Requires good filtration"]
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
        
        "description": "Balanced performance with 22° flute angle for optimal water distribution. Good fouling resistance with moderate heat transfer.",
        "applications": ["General purpose cooling", "HVAC systems", "Industrial process cooling"],
        "limitations": ["Moderate efficiency", "Standard performance profile"]
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
        
        "description": "31° flute angle enhances heat transfer by increasing water turbulence. Good balance between performance and pressure drop.",
        "applications": ["Process cooling", "Power plants", "High efficiency requirements"],
        "limitations": ["Requires good water treatment", "Higher initial cost"]
    },
    
    "XF125SS": {
        "name": "Brentwood XF125SS",
        "surface_area": 157.5,
        "sheet_spacing": 19.0,
        "flute_angle": 27,
        "channel_depth": 11.0,
        "channel_width": 16.5,
        "hydraulic_diameter": 10.5,
        "free_area_fraction": 0.91,
        "water_passage_area": 0.82,
        "material_thickness_options": [0.13, 0.15],
        "dry_weight_range": [35.2, 76.9],
        "water_film_thickness": 0.8,
        "max_water_loading": 16,
        "min_water_loading": 5,
        "recommended_air_velocity": 2.3,
        "max_air_velocity": 2.8,
        "fouling_factor": 0.92,
        
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5],
            "Ka_L": [2.05, 1.74, 1.48, 1.27, 1.09],
            "delta_P_base": [26, 33, 43, 57, 74]
        },
        
        "description": "Staggered sheet configuration for specific flow patterns. Specialized for retrofit or custom applications.",
        "applications": ["Retrofit projects", "Specific flow distribution", "Custom installations"],
        "limitations": ["Limited size options", "Specialized application"]
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
        
        "description": "Wide sheet spacing and large water channels provide excellent fouling resistance and low pressure drop. Ideal for challenging water conditions.",
        "applications": ["Dirty water applications", "High air volume systems", "Low pressure drop requirements"],
        "limitations": ["Lower thermal efficiency", "Larger volume required"]
    }
}

# ============================================================================
# ENHANCED CALCULATION FUNCTIONS USING GEOMETRIC PARAMETERS
# ============================================================================

def calculate_hydraulic_properties(fill_data, water_loading, air_face_velocity):
    """
    Calculate hydraulic properties based on fill geometry
    """
    # Convert water loading from m³/h·m² to m/s
    water_velocity_ms = (water_loading / 3.6) / fill_data["water_passage_area"]
    
    # Water film Reynolds number
    water_viscosity = 1e-6  # m²/s at 30°C
    film_reynolds = (water_velocity_ms * fill_data["water_film_thickness"] * 1e-3) / water_viscosity
    
    # Air side Reynolds number
    air_viscosity = 1.5e-5  # m²/s at 30°C
    air_reynolds = (air_face_velocity * fill_data["hydraulic_diameter"] * 1e-3) / air_viscosity
    
    # Water film Weber number (surface tension effects)
    surface_tension = 0.072  # N/m at 30°C
    water_density = 995  # kg/m³ at 30°C
    weber_number = (water_density * (water_velocity_ms**2) * 
                   fill_data["water_film_thickness"] * 1e-3) / surface_tension
    
    return {
        "water_velocity": water_velocity_ms,  # m/s
        "film_reynolds": film_reynolds,
        "air_reynolds": air_reynolds,
        "weber_number": weber_number,
        "film_thickness_actual": fill_data["water_film_thickness"]  # mm
    }

def assess_fouling_risk(fill_data, hydraulic_props, water_quality="good"):
    """
    Assess fouling risk based on fill geometry and operating conditions
    """
    risk_score = 0
    
    # Factors increasing fouling risk
    # 1. Small hydraulic diameter (higher risk)
    if fill_data["hydraulic_diameter"] < 10:
        risk_score += 2
    elif fill_data["hydraulic_diameter"] < 12:
        risk_score += 1
    
    # 2. Low water velocity (sedimentation)
    if hydraulic_props["water_velocity"] < 0.05:  # m/s
        risk_score += 3
    elif hydraulic_props["water_velocity"] < 0.1:
        risk_score += 1
    
    # 3. Low Reynolds number (laminar flow promotes fouling)
    if hydraulic_props["film_reynolds"] < 500:
        risk_score += 2
    elif hydraulic_props["film_reynolds"] < 1000:
        risk_score += 1
    
    # Water quality factor
    water_quality_factors = {
        "excellent": 0.5,
        "good": 1.0,
        "fair": 1.5,
        "poor": 2.0,
        "very_poor": 3.0
    }
    
    risk_score *= water_quality_factors.get(water_quality, 1.0)
    
    # Risk categories
    if risk_score < 3:
        risk_level = "Low"
        recommendation = "Normal water treatment sufficient"
    elif risk_score < 6:
        risk_level = "Moderate"
        recommendation = "Enhanced filtration recommended"
    elif risk_score < 9:
        risk_level = "High"
        recommendation = "Regular chemical treatment and filtration required"
    else:
        risk_level = "Very High"
        recommendation = "Consider alternative fill type or extensive treatment"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "critical_factors": [
            f"Hydraulic diameter: {fill_data['hydraulic_diameter']:.1f} mm",
            f"Water velocity: {hydraulic_props['water_velocity']:.3f} m/s",
            f"Film Re: {hydraulic_props['film_reynolds']:.0f}"
        ]
    }

def estimate_heat_transfer_from_geometry(fill_data, hydraulic_props):
    """
    Estimate heat transfer coefficient enhancement factors from geometry
    """
    # Base heat transfer factor (normalized)
    base_factor = fill_data["surface_area"] / 200  # Normalize to 200 m²/m³
    
    # Flute angle enhancement (higher angle = more turbulence)
    angle_factor = 1.0 + (fill_data["flute_angle"] - 25) * 0.02
    
    # Water film Reynolds enhancement
    if hydraulic_props["film_reynolds"] > 1500:
        re_factor = 1.2  # Turbulent enhancement
    elif hydraulic_props["film_reynolds"] > 500:
        re_factor = 1.0  # Transitional
    else:
        re_factor = 0.8  # Laminar reduction
    
    # Combined enhancement factor
    enhancement_factor = base_factor * angle_factor * re_factor * fill_data["fouling_factor"]
    
    return {
        "base_factor": base_factor,
        "angle_factor": angle_factor,
        "reynolds_factor": re_factor,
        "total_enhancement": enhancement_factor,
        "estimated_relative_performance": enhancement_factor
    }

# ============================================================================
# ENHANCED MERKEL SOLVER WITH GEOMETRIC CORRECTIONS
# ============================================================================

def solve_merkel_with_geometry(L, G, T_hot, T_cold_target, Twb, fill_type, 
                               fill_depth, face_area, water_quality="good"):
    """
    Enhanced Merkel solver incorporating geometric parameters
    """
    fill_data = BRENTWOOD_FILLS_ENHANCED[fill_type]
    
    # Basic calculations
    L_over_G = L / G
    Ka_over_L = np.interp(L_over_G, 
                         fill_data["performance_data"]["L_G"], 
                         fill_data["performance_data"]["Ka_L"])
    
    Ka = Ka_over_L * L  # kW/°C
    
    # Air properties
    air_density = 1.2  # kg/m³
    air_flow_volumetric = G / air_density
    air_face_velocity = air_flow_volumetric / face_area
    
    # Water loading
    water_loading = (L * 3.6) / face_area
    
    # Calculate hydraulic properties
    hydraulic_props = calculate_hydraulic_properties(fill_data, water_loading, air_face_velocity)
    
    # Estimate heat transfer enhancement
    ht_enhancement = estimate_heat_transfer_from_geometry(fill_data, hydraulic_props)
    
    # Adjust Ka based on geometric enhancement
    Ka_enhanced = Ka * ht_enhancement["total_enhancement"]
    
    # Assess fouling risk
    fouling_assessment = assess_fouling_risk(fill_data, hydraulic_props, water_quality)
    
    # Pressure drop calculation with geometric correction
    delta_P_base = np.interp(L_over_G, 
                            fill_data["performance_data"]["L_G"], 
                            fill_data["performance_data"]["delta_P_base"])
    
    # Velocity correction (ΔP ∝ V²)
    velocity_factor = (air_face_velocity / fill_data["recommended_air_velocity"]) ** 1.8
    
    # Hydraulic diameter correction (ΔP ∝ 1/D_h)
    diameter_factor = (12 / fill_data["hydraulic_diameter"]) ** 0.5
    
    fill_pressure_drop = delta_P_base * velocity_factor * diameter_factor * fill_depth
    total_static_pressure = fill_pressure_drop * 1.35
    
    # Check operating limits
    operating_warnings = []
    
    if water_loading > fill_data["max_water_loading"]:
        operating_warnings.append(f"⚠️ Water loading ({water_loading:.1f} m³/h·m²) exceeds maximum ({fill_data['max_water_loading']} m³/h·m²)")
    
    if water_loading < fill_data["min_water_loading"]:
        operating_warnings.append(f"⚠️ Water loading ({water_loading:.1f} m³/h·m²) below minimum ({fill_data['min_water_loading']} m³/h·m²)")
    
    if air_face_velocity > fill_data["max_air_velocity"]:
        operating_warnings.append(f"⚠️ Air face velocity ({air_face_velocity:.2f} m/s) exceeds maximum ({fill_data['max_air_velocity']} m/s)")
    
    # Simplified Merkel calculation (using enhanced Ka)
    Cp = 4.186  # kJ/kg°C
    NTU = (Ka_enhanced / (L * Cp)) * fill_depth
    
    # Achieved cold water temperature (simplified)
    T_cold_achieved = Twb + (T_hot - Twb) * np.exp(-NTU)
    Q_achieved = L * Cp * (T_hot - T_cold_achieved)
    
    return {
        # Basic results
        "fill_type": fill_type,
        "fill_name": fill_data["name"],
        "T_cold_achieved": T_cold_achieved,
        "T_cold_target": T_cold_target,
        "Q_achieved": Q_achieved,
        "approach": T_cold_achieved - Twb,
        "cooling_range": T_hot - T_cold_achieved,
        
        # Hydraulic properties
        "water_loading": water_loading,
        "air_face_velocity": air_face_velocity,
        "hydraulic_props": hydraulic_props,
        
        # Geometric enhancements
        "ht_enhancement": ht_enhancement,
        "Ka_original": Ka,
        "Ka_enhanced": Ka_enhanced,
        "NTU": NTU,
        
        # Pressure and flow
        "air_flow_volumetric": air_flow_volumetric,
        "total_static_pressure": total_static_pressure,
        "L_over_G": L_over_G,
        
        # Assessments
        "fouling_assessment": fouling_assessment,
        "operating_warnings": operating_warnings,
        
        # Geometry info
        "surface_area": fill_data["surface_area"],
        "hydraulic_diameter": fill_data["hydraulic_diameter"],
        "flute_angle": fill_data["flute_angle"],
        "free_area_fraction": fill_data["free_area_fraction"]
    }

# ============================================================================
# VISUALIZATION FUNCTIONS FOR GEOMETRY
# ============================================================================

def plot_fill_geometry(fill_data, ax=None):
    """Create a visualization of fill geometry"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw fill sheet with channels
    sheet_height = fill_data["sheet_spacing"]
    channel_depth = fill_data["channel_depth"]
    channel_width = fill_data["channel_width"]
    
    # Create simplified representation
    x_positions = np.linspace(0, 50, 10)
    
    for i, x in enumerate(x_positions):
        # Sheet
        ax.plot([x, x + channel_width], 
                [0, fill_data["flute_angle"] / 30 * channel_width], 
                'k-', linewidth=fill_data.get("material_thickness_options", [0.3])[0] * 10)
        
        # Water channel
        if i < len(x_positions) - 1:
            ax.fill_between([x + channel_width/4, x + 3*channel_width/4],
                           [0.2, 0.2],
                           [channel_depth/5, channel_depth/5],
                           alpha=0.3, color='blue', label='Water Channel' if i == 0 else "")
    
    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Height (mm)")
    ax.set_title(f"{fill_data['name']} - Geometry")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    # Add annotations
    ax.text(5, sheet_height + 2, f"Sheet Spacing: {sheet_height} mm", fontsize=9)
    ax.text(5, sheet_height + 5, f"Flute Angle: {fill_data['flute_angle']}°", fontsize=9)
    ax.text(5, sheet_height + 8, f"Channel Depth: {channel_depth} mm", fontsize=9)
    
    return ax

def create_geometry_comparison_chart(fills_data):
    """Create comparison chart of geometric parameters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    fill_names = [d["name"] for d in fills_data]
    
    # Plot 1: Surface Area
    surface_areas = [d["surface_area"] for d in fills_data]
    bars1 = axes[0, 0].bar(fill_names, surface_areas, color='skyblue')
    axes[0, 0].set_ylabel("Surface Area (m²/m³)")
    axes[0, 0].set_title("Surface Area Comparison")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Hydraulic Diameter
    hyd_diameters = [d["hydraulic_diameter"] for d in fills_data]
    bars2 = axes[0, 1].bar(fill_names, hyd_diameters, color='lightgreen')
    axes[0, 1].set_ylabel("Hydraulic Diameter (mm)")
    axes[0, 1].set_title("Channel Size Comparison")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Flute Angle
    flute_angles = [d["flute_angle"] for d in fills_data]
    bars3 = axes[0, 2].bar(fill_names, flute_angles, color='gold')
    axes[0, 2].set_ylabel("Flute Angle (°)")
    axes[0, 2].set_title("Flute Angle Comparison")
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Free Area Fraction
    free_areas = [d["free_area_fraction"] for d in fills_data]
    bars4 = axes[1, 0].bar(fill_names, free_areas, color='salmon')
    axes[1, 0].set_ylabel("Free Area Fraction")
    axes[1, 0].set_title("Air Flow Area Comparison")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim([0.8, 1.0])
    
    # Plot 5: Fouling Factor
    fouling_factors = [d["fouling_factor"] for d in fills_data]
    bars5 = axes[1, 1].bar(fill_names, fouling_factors, color='purple')
    axes[1, 1].set_ylabel("Fouling Resistance")
    axes[1, 1].set_title("Fouling Resistance Comparison")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim([0.8, 1.0])
    
    # Plot 6: Recommended Water Loading
    water_loadings = [d["max_water_loading"] for d in fills_data]
    bars6 = axes[1, 2].bar(fill_names, water_loadings, color='orange')
    axes[1, 2].set_ylabel("Max Water Loading (m³/h·m²)")
    axes[1, 2].set_title("Maximum Water Loading")
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

# ============================================================================
# ENHANCED STREAMLIT INTERFACE
# ============================================================================

def main_enhanced():
    """Enhanced main function with geometric parameters"""
    
    # Check password (simplified version)
    if "authenticated" not in st.session_state:
        password = st.text_input("Enter Password", type="password")
        if password == "Semaanju":
            st.session_state.authenticated = True
            st.rerun()
        else:
            if password:
                st.error("Incorrect password")
            return
    
    st.set_page_config(layout="wide")
    st.title("🌊 Cooling Tower Design - Enhanced with Geometric Parameters")
    
    # Sidebar
    with st.sidebar:
        st.header("📥 Design Inputs")
        
        # Simplified inputs for demonstration
        L = st.number_input("Water Flow (kg/s)", value=100.0)
        T_hot = st.number_input("Hot Water In (°C)", value=37.0)
        T_cold_target = st.number_input("Target Cold Water Out (°C)", value=32.0)
        Twb = st.number_input("Wet Bulb (°C)", value=28.0)
        L_over_G = st.slider("L/G Ratio", 0.5, 2.0, 1.25, 0.05)
        fill_depth = st.slider("Fill Depth (m)", 0.3, 2.0, 0.6, 0.1)
        face_area = st.slider("Face Area (m²)", 10.0, 100.0, 36.94, 5.0)
        
        # Water quality
        water_quality = st.selectbox(
            "Water Quality",
            ["excellent", "good", "fair", "poor", "very_poor"],
            index=1
        )
        
        # Fill selection
        fill_options = list(BRENTWOOD_FILLS_ENHANCED.keys())
        selected_fills = st.multiselect(
            "Select Fills",
            fill_options,
            default=["XF75", "XF125", "XF3000"]
        )
        
        run_calc = st.button("Run Enhanced Analysis", type="primary")
    
    if run_calc:
        G = L / L_over_G
        results = []
        fills_data = []
        
        for fill_type in selected_fills:
            result = solve_merkel_with_geometry(
                L, G, T_hot, T_cold_target, Twb, fill_type,
                fill_depth, face_area, water_quality
            )
            results.append(result)
            fills_data.append(BRENTWOOD_FILLS_ENHANCED[fill_type])
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        # 1. Performance Comparison
        st.header("📊 Performance Results")
        cols = st.columns(len(results))
        for idx, (col, result) in enumerate(zip(cols, results)):
            with col:
                st.subheader(result["fill_name"])
                st.metric("Cold Water", f"{result['T_cold_achieved']:.2f}°C")
                st.metric("Heat Rejection", f"{result['Q_achieved']:.0f} kW")
                st.metric("Approach", f"{result['approach']:.2f}°C")
                st.metric("Static Pressure", f"{result['total_static_pressure']:.0f} Pa")
                
                # Display warnings if any
                for warning in result["operating_warnings"]:
                    st.warning(warning)
        
        # 2. Geometry Comparison Chart
        st.header("📐 Geometric Parameter Comparison")
        fig = create_geometry_comparison_chart(fills_data)
        st.pyplot(fig)
        
        # 3. Hydraulic Properties
        st.header("💧 Hydraulic Analysis")
        for result in results:
            with st.expander(f"{result['fill_name']} - Hydraulic Properties"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Flow Properties:**")
                    st.write(f"- Water Loading: {result['water_loading']:.1f} m³/h·m²")
                    st.write(f"- Air Face Velocity: {result['air_face_velocity']:.2f} m/s")
                    st.write(f"- Water Velocity in Channels: {result['hydraulic_props']['water_velocity']:.3f} m/s")
                    st.write(f"- Film Reynolds Number: {result['hydraulic_props']['film_reynolds']:.0f}")
                    st.write(f"- Air Reynolds Number: {result['hydraulic_props']['air_reynolds']:.0f}")
                
                with col2:
                    st.markdown("**Geometric Parameters:**")
                    st.write(f"- Surface Area: {result['surface_area']} m²/m³")
                    st.write(f"- Hydraulic Diameter: {result['hydraulic_diameter']:.1f} mm")
                    st.write(f"- Flute Angle: {result['flute_angle']}°")
                    st.write(f"- Free Area Fraction: {result['free_area_fraction']:.2f}")
                    st.write(f"- Estimated Film Thickness: {BRENTWOOD_FILLS_ENHANCED[result['fill_type']]['water_film_thickness']} mm")
        
        # 4. Fouling Risk Assessment
        st.header("⚠️ Fouling Risk Assessment")
        for result in results:
            assessment = result["fouling_assessment"]
            with st.expander(f"{result['fill_name']} - {assessment['risk_level']} Risk"):
                st.metric("Risk Score", f"{assessment['risk_score']:.1f}", 
                         delta=assessment['risk_level'])
                st.info(f"**Recommendation**: {assessment['recommendation']}")
                
                st.write("**Critical Factors:**")
                for factor in assessment["critical_factors"]:
                    st.write(f"- {factor}")
        
        # 5. Heat Transfer Enhancement Analysis
        st.header("🔥 Heat Transfer Enhancement")
        for result in results:
            with st.expander(f"{result['fill_name']} - Enhancement Factors"):
                enhancement = result["ht_enhancement"]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Enhancement Factors:**")
                    st.write(f"- Base Factor (Surface Area): {enhancement['base_factor']:.3f}")
                    st.write(f"- Flute Angle Factor: {enhancement['angle_factor']:.3f}")
                    st.write(f"- Reynolds Number Factor: {enhancement['reynolds_factor']:.3f}")
                    st.write(f"- Fouling Factor: {BRENTWOOD_FILLS_ENHANCED[result['fill_type']]['fouling_factor']:.3f}")
                    st.write(f"- **Total Enhancement**: {enhancement['total_enhancement']:.3f}")
                
                with col2:
                    st.markdown("**Heat Transfer Coefficients:**")
                    st.write(f"- Ka (from curves): {result['Ka_original']:.1f} kW/°C")
                    st.write(f"- Ka (enhanced): {result['Ka_enhanced']:.1f} kW/°C")
                    st.write(f"- Enhancement: {((result['Ka_enhanced']/result['Ka_original'])-1)*100:.1f}%")
                    st.write(f"- NTU: {result['NTU']:.3f}")
        
        # 6. Fill Geometry Visualization
        st.header("📐 Fill Geometry Visualization")
        selected_for_viz = st.selectbox(
            "Select fill to visualize geometry:",
            selected_fills,
            format_func=lambda x: BRENTWOOD_FILLS_ENHANCED[x]["name"]
        )
        
        fig_viz, ax_viz = plt.subplots(figsize=(10, 6))
        plot_fill_geometry(BRENTWOOD_FILLS_ENHANCED[selected_for_viz], ax_viz)
        st.pyplot(fig_viz)
        
        # 7. Detailed Fill Specifications
        st.header("📋 Fill Specifications")
        for fill_type in selected_fills:
            fill_data = BRENTWOOD_FILLS_ENHANCED[fill_type]
            with st.expander(f"{fill_data['name']} - Complete Specifications"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Geometric Parameters:**")
                    st.write(f"- Surface Area: {fill_data['surface_area']} m²/m³")
                    st.write(f"- Sheet Spacing: {fill_data['sheet_spacing']} mm")
                    st.write(f"- Flute Angle: {fill_data['flute_angle']}°")
                    st.write(f"- Channel Depth: {fill_data['channel_depth']} mm")
                    st.write(f"- Channel Width: {fill_data['channel_width']} mm")
                    st.write(f"- Hydraulic Diameter: {fill_data['hydraulic_diameter']} mm")
                
                with col2:
                    st.markdown("**Operational Limits:**")
                    st.write(f"- Min Water Loading: {fill_data['min_water_loading']} m³/h·m²")
                    st.write(f"- Max Water Loading: {fill_data['max_water_loading']} m³/h·m²")
                    st.write(f"- Recommended Air Velocity: {fill_data['recommended_air_velocity']} m/s")
                    st.write(f"- Max Air Velocity: {fill_data['max_air_velocity']} m/s")
                    st.write(f"- Free Area Fraction: {fill_data['free_area_fraction']}")
                    st.write(f"- Fouling Factor: {fill_data['fouling_factor']}")
                
                with col3:
                    st.markdown("**Construction:**")
                    st.write(f"- Material Thickness Options: {', '.join(map(str, fill_data['material_thickness_options']))} mm")
                    st.write(f"- Dry Weight Range: {fill_data['dry_weight_range'][0]} - {fill_data['dry_weight_range'][1]} kg/m³")
                    st.write(f"- Water Film Thickness: {fill_data['water_film_thickness']} mm")
                    st.write(f"- Water Passage Area: {fill_data['water_passage_area']}")
                
                st.markdown("**Description:**")
                st.write(fill_data['description'])
                
                st.markdown("**Applications:**")
                for app in fill_data.get('applications', []):
                    st.write(f"- {app}")
                
                st.markdown("**Limitations:**")
                for limit in fill_data.get('limitations', []):
                    st.write(f"- {limit}")
        
        # 8. Design Recommendations
        st.header("🎯 Design Recommendations")
        
        # Find best fill for different criteria
        if len(results) > 1:
            metrics = {
                "Highest Efficiency": max(results, key=lambda x: x["surface_area"]),
                "Lowest Fouling Risk": min(results, key=lambda x: x["fouling_assessment"]["risk_score"]),
                "Lowest Pressure Drop": min(results, key=lambda x: x["total_static_pressure"]),
                "Best Cold Water": min(results, key=lambda x: x["T_cold_achieved"])
            }
            
            cols = st.columns(len(metrics))
            for idx, (criterion, result) in enumerate(metrics.items()):
                with cols[idx]:
                    st.metric(criterion, result["fill_name"])
                    if criterion == "Highest Efficiency":
                        st.write(f"Surface Area: {result['surface_area']} m²/m³")
                    elif criterion == "Lowest Fouling Risk":
                        st.write(f"Risk: {result['fouling_assessment']['risk_level']}")
                    elif criterion == "Lowest Pressure Drop":
                        st.write(f"ΔP: {result['total_static_pressure']:.0f} Pa")
                    elif criterion == "Best Cold Water":
                        st.write(f"Temp: {result['T_cold_achieved']:.2f}°C")

if __name__ == "__main__":
    main_enhanced()