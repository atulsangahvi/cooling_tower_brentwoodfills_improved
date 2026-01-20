# cooling_tower_design_fixed.py
# Fixed Cooling Tower Design Tool - No external dependencies

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# BRENTWOOD FILL DATABASE
# ============================================================================

BRENTWOOD_FILLS = {
    "CF1200": {
        "name": "Brentwood ACCU-PAK CF1200",
        "surface_area": 226,
        "max_water_loading": 14,
        "min_water_loading": 6,
        "recommended_water_loading": 8,
        "max_air_velocity": 2.8,
        "recommended_air_velocity": 2.2,
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
            "Ka_L": [1.9, 1.6, 1.35, 1.15, 0.98, 0.85, 0.75, 0.67, 0.60],
            "delta_P_base": [45, 58, 75, 96, 122, 152, 186, 225, 268]
        },
    },
    
    "XF75": {
        "name": "Brentwood XF75",
        "surface_area": 226,
        "max_water_loading": 15,
        "min_water_loading": 5,
        "recommended_water_loading": 8,
        "max_air_velocity": 3.0,
        "recommended_air_velocity": 2.5,
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.3, 1.95, 1.65, 1.4, 1.2, 1.05, 0.92],
            "delta_P_base": [35, 45, 60, 80, 105, 135, 170]
        },
    },
    
    "XF3000": {
        "name": "Brentwood XF3000",
        "surface_area": 102,
        "max_water_loading": 25,
        "min_water_loading": 8,
        "recommended_water_loading": 15,
        "max_air_velocity": 3.2,
        "recommended_air_velocity": 2.6,
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [1.7, 1.45, 1.25, 1.08, 0.94, 0.83, 0.74],
            "delta_P_base": [18, 23, 30, 39, 51, 65, 81]
        },
    }
}

TOWER_TYPES = {
    "crossflow": {"name": "Crossflow", "fill_utilization": 0.85, "pressure_factor": 1.0},
    "counterflow_induced": {"name": "Counterflow (Induced Draft)", "fill_utilization": 0.95, "pressure_factor": 1.3},
    "counterflow_forced": {"name": "Counterflow (Forced Draft)", "fill_utilization": 0.92, "pressure_factor": 1.2}
}

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def interpolate_value(x, x_points, y_points):
    """Simple linear interpolation"""
    if x <= x_points[0]:
        return y_points[0]
    if x >= x_points[-1]:
        return y_points[-1]
    
    for i in range(len(x_points) - 1):
        if x_points[i] <= x <= x_points[i + 1]:
            x1, x2 = x_points[i], x_points[i + 1]
            y1, y2 = y_points[i], y_points[i + 1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    return y_points[-1]

def calculate_KaL(fill_data, L_over_G, tower_type):
    """Calculate Ka/L for given fill and L/G ratio"""
    Ka_over_L = interpolate_value(
        L_over_G,
        fill_data["performance_data"]["L_G"],
        fill_data["performance_data"]["Ka_L"]
    )
    
    # Adjust for tower type
    if tower_type.startswith("counterflow"):
        efficiency_factor = TOWER_TYPES[tower_type]["fill_utilization"] / 0.85
        Ka_over_L *= efficiency_factor
    
    return Ka_over_L

def solve_cooling_tower(L, G, T_hot, T_cold_target, Twb, Tdb, fill_type, tower_type, fill_depth, face_area, altitude=0):
    """Main calculation function"""
    fill_data = BRENTWOOD_FILLS[fill_type]
    tower_data = TOWER_TYPES[tower_type]
    
    # Basic calculations
    L_over_G = L / G
    Ka_over_L = calculate_KaL(fill_data, L_over_G, tower_type)
    NTU = Ka_over_L * fill_depth * (1.05 if tower_type.startswith("counterflow") else 1.0)
    
    # Air properties (simplified)
    P_atm = 101.325 * (1 - 0.0000225577 * altitude) ** 5.25588
    air_density = 1.2 * (P_atm / 101.325) * (288.15 / (Tdb + 273.15))
    
    # Flow calculations
    water_loading = (L * 3.6) / face_area  # m³/h·m²
    air_flow_volumetric = G / air_density  # m³/s
    air_face_velocity = air_flow_volumetric / face_area  # m/s
    
    # Temperature calculations
    approach_factor = 0.95 if tower_type.startswith("counterflow") else 1.0
    T_cold_achieved = Twb + (T_hot - Twb) * np.exp(-NTU * approach_factor)
    T_cold_achieved = max(T_cold_achieved, Twb + 0.5)
    T_cold_achieved = min(T_cold_achieved, T_hot - 0.5)
    
    Q_achieved = L * 4.186 * (T_hot - T_cold_achieved)
    
    # Geometry
    fill_volume = face_area * fill_depth
    total_surface_area = fill_volume * fill_data["surface_area"]
    
    # Pressure drop
    delta_P_base = interpolate_value(
        L_over_G,
        fill_data["performance_data"]["L_G"],
        fill_data["performance_data"]["delta_P_base"]
    )
    velocity_factor = (air_face_velocity / 2.5) ** 2
    fill_pressure_drop = delta_P_base * velocity_factor * fill_depth * tower_data["pressure_factor"]
    total_static_pressure = fill_pressure_drop + 30  # Additional losses
    
    # Fan power
    fan_efficiency = 0.78
    fan_power = (air_flow_volumetric * total_static_pressure) / (fan_efficiency * 1000)
    
    # Check limits
    max_water_loading = fill_data["max_water_loading"]
    min_water_loading = fill_data["min_water_loading"]
    max_air_velocity = fill_data["max_air_velocity"]
    
    water_status = "❌ Exceeds Max" if water_loading > max_water_loading else \
                   "⚠️ Below Min" if water_loading < min_water_loading else "✅ Within Limits"
    
    air_status = "❌ Exceeds Max" if air_face_velocity > max_air_velocity else "✅ Within Limits"
    
    # Warnings
    warnings = []
    if water_loading > max_water_loading:
        warnings.append(f"Water loading exceeds maximum ({max_water_loading} m³/h·m²)")
    if air_face_velocity > max_air_velocity:
        warnings.append(f"Air velocity exceeds maximum ({max_air_velocity} m/s)")
    
    # Relative humidity (simplified)
    RH = 100 * (np.exp(17.27 * Twb / (Twb + 237.3)) / np.exp(17.27 * Tdb / (Tdb + 237.3)))
    
    return {
        "fill_name": fill_data["name"],
        "tower_name": tower_data["name"],
        "T_cold_achieved": T_cold_achieved,
        "T_cold_target": T_cold_target,
        "water_loading": water_loading,
        "air_face_velocity": air_face_velocity,
        "water_status": water_status,
        "air_status": air_status,
        "max_water_loading": max_water_loading,
        "min_water_loading": min_water_loading,
        "recommended_water_loading": fill_data["recommended_water_loading"],
        "max_air_velocity": max_air_velocity,
        "recommended_air_velocity": fill_data["recommended_air_velocity"],
        "Q_achieved": Q_achieved,
        "approach": T_cold_achieved - Twb,
        "range": T_hot - T_cold_achieved,
        "L_over_G": L_over_G,
        "NTU": NTU,
        "Ka_over_L": Ka_over_L,
        "total_surface_area": total_surface_area,
        "fan_power": fan_power,
        "total_static_pressure": total_static_pressure,
        "face_area": face_area,
        "fill_depth": fill_depth,
        "fill_volume": fill_volume,
        "air_flow_volumetric": air_flow_volumetric,
        "RH": RH,
        "warnings": warnings
    }

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    if not check_password():
        st.stop()
    
    st.set_page_config(page_title="Cooling Tower Design", layout="wide")
    st.title("🌊 Cooling Tower Design Tool")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Design Inputs")
        
        # Temperatures
        col1, col2 = st.columns(2)
        with col1:
            T_hot = st.number_input("Hot Water In (°C)", value=40.0, step=0.5, format="%.1f")
        with col2:
            T_cold_target = st.number_input("Cold Water Target (°C)", value=35.0, step=0.5, format="%.1f")
        
        col3, col4 = st.columns(2)
        with col3:
            Twb = st.number_input("Wet Bulb (°C)", value=31.0, step=0.5, format="%.1f")
        with col4:
            Tdb = st.number_input("Dry Bulb (°C)", value=45.0, step=0.5, format="%.1f")
        
        # Flow rates
        L = st.number_input("Water Flow (kg/s)", value=103.3, step=1.0, format="%.1f")
        G = st.number_input("Air Flow (kg/s)", value=44.72, step=1.0, format="%.2f")
        
        # Geometry
        fill_depth = st.number_input("Fill Depth (m)", value=0.750, step=0.050, format="%.3f")
        
        st.subheader("Tower Shape")
        tower_shape = st.radio("Shape:", ["Rectangle", "Round"], horizontal=True)
        
        if tower_shape == "Rectangle":
            col1, col2 = st.columns(2)
            with col1:
                length = st.number_input("Length (m)", value=3.6, step=0.1, format="%.1f")
            with col2:
                width = st.number_input("Width (m)", value=3.6, step=0.1, format="%.1f")
            face_area = length * width
        else:
            diameter = st.number_input("Diameter (m)", value=4.06, step=0.1, format="%.2f")
            face_area = math.pi * (diameter / 2) ** 2
        
        st.info(f"**Face Area:** {face_area:.2f} m²")
        
        # Tower type
        tower_type = st.selectbox("Tower Type:", list(TOWER_TYPES.keys()),
                                 format_func=lambda x: TOWER_TYPES[x]["name"])
        
        # Fill selection
        fills = list(BRENTWOOD_FILLS.keys())
        selected_fills = st.multiselect("Select Fills:", fills, default=["CF1200", "XF75", "XF3000"],
                                       format_func=lambda x: BRENTWOOD_FILLS[x]["name"])
        
        altitude = st.number_input("Altitude (m ASL)", value=0, step=100)
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_button and selected_fills:
        if T_hot <= T_cold_target:
            st.error("Hot water must be greater than cold water target")
            st.stop()
        
        # Calculate results
        results = []
        for fill in selected_fills:
            result = solve_cooling_tower(
                L, G, T_hot, T_cold_target, Twb, Tdb, fill,
                tower_type, fill_depth, face_area, altitude
            )
            results.append(result)
        
        # Display metrics
        st.header("📊 Performance Results")
        cols = st.columns(len(selected_fills))
        
        for idx, (col, result) in enumerate(zip(cols, results)):
            with col:
                st.subheader(result["fill_name"])
                
                # Temperature
                temp_emoji = "✅" if result["T_cold_achieved"] <= result["T_cold_target"] else "❌"
                st.metric(f"{temp_emoji} Cold Water", 
                         f"{result['T_cold_achieved']:.2f}°C",
                         delta=f"{result['T_cold_achieved'] - result['T_cold_target']:.2f}°C")
                
                # Water loading
                water_emoji = result["water_status"][0]
                st.metric(f"{water_emoji} Water Loading", 
                         f"{result['water_loading']:.1f} m³/h·m²",
                         delta=f"Max: {result['max_water_loading']} m³/h·m²")
                
                # Air velocity
                air_emoji = result["air_status"][0]
                st.metric(f"{air_emoji} Air Velocity", 
                         f"{result['air_face_velocity']:.2f} m/s",
                         delta=f"Max: {result['max_air_velocity']} m/s")
                
                st.metric("Fan Power", f"{result['fan_power']:.2f} kW")
                st.metric("Heat Rejection", f"{result['Q_achieved']:.0f} kW")
        
        # Critical issues
        critical_issues = []
        for result in results:
            if result["water_loading"] > result["max_water_loading"]:
                critical_issues.append(f"**{result['fill_name']}**: Water loading {result['water_loading']:.1f} > {result['max_water_loading']} m³/h·m²")
            if result["air_face_velocity"] > result["max_air_velocity"]:
                critical_issues.append(f"**{result['fill_name']}**: Air velocity {result['air_face_velocity']:.2f} > {result['max_air_velocity']} m/s")
        
        if critical_issues:
            st.error("🚨 **Manufacturer Limits Exceeded:**")
            for issue in critical_issues:
                st.write(f"- {issue}")
            
            st.warning("**Recommended Actions:**")
            st.write("1. Reduce water flow or increase face area")
            st.write("2. Reduce air flow or increase face area")
            st.write("3. Select XF3000 for higher water loading capacity")
        
        # Detailed table
        st.header("📋 Detailed Results")
        table_data = []
        for result in results:
            table_data.append({
                "Fill": result["fill_name"],
                "Cold Water (°C)": f"{result['T_cold_achieved']:.2f}",
                "Water Loading (m³/h·m²)": f"{result['water_loading']:.1f} {result['water_status']}",
                "Water Limits": f"{result['min_water_loading']}-{result['max_water_loading']}",
                "Air Velocity (m/s)": f"{result['air_face_velocity']:.2f} {result['air_status']}",
                "Air Limit": f"≤{result['max_air_velocity']}",
                "Fan Power (kW)": f"{result['fan_power']:.2f}",
                "Heat Rejection (kW)": f"{result['Q_achieved']:.0f}",
                "Approach (°C)": f"{result['approach']:.2f}"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        st.header("📈 Performance Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cold water temperatures
        fill_names = [r["fill_name"] for r in results]
        cold_temps = [r["T_cold_achieved"] for r in results]
        
        bars1 = ax1.bar(fill_names, cold_temps, color=['red' if t > T_cold_target else 'green' for t in cold_temps])
        ax1.axhline(y=T_cold_target, color='blue', linestyle='--', label='Target')
        ax1.set_ylabel('Cold Water Temperature (°C)')
        ax1.set_title('Thermal Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # Water loading
        water_loads = [r["water_loading"] for r in results]
        max_limits = [r["max_water_loading"] for r in results]
        
        x = np.arange(len(fill_names))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, water_loads, width, label='Actual', color='blue')
        bars3 = ax2.bar(x + width/2, max_limits, width, label='Max Limit', color='red', alpha=0.6)
        ax2.set_ylabel('Water Loading (m³/h·m²)')
        ax2.set_title('Water Loading vs Limits')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fill_names, rotation=45)
        ax2.legend()
        
        st.pyplot(fig)
        
    elif run_button and not selected_fills:
        st.warning("Please select at least one fill type.")
    else:
        # Welcome screen
        st.markdown("""
        ## Cooling Tower Design Tool
        
        **Features:**
        - CF1200, XF75, XF3000 fill support
        - Counterflow and crossflow tower types
        - Manufacturer limits display
        - Water loading analysis
        
        **How to use:**
        1. Enter temperatures and flow rates in sidebar
        2. Select tower shape and dimensions
        3. Choose fills to compare
        4. Click "Run Analysis"
        
        **Manufacturer Limits:**
        - CF1200: Water loading 6-14 m³/h·m²
        - XF75: Water loading 5-15 m³/h·m²  
        - XF3000: Water loading 8-25 m³/h·m²
        
        *Configure inputs in the sidebar and run analysis.*
        """)

if __name__ == "__main__":
    main()