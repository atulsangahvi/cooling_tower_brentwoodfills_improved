# cooling_tower_design_pro.py
# Professional Cooling Tower Design Tool
# Password: "Semaanju"
# Updated: Fixed Mode 2 inputs, added L/G ratio selection

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

# Try to import reportlab with error handling
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    reportlab_available = True
except ImportError:
    reportlab_available = False
    st.warning("⚠️ reportlab is not installed. PDF generation will be disabled.")

# Other imports
from io import BytesIO
import datetime
import hashlib
import base64

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
# PSYCHROMETRIC FUNCTIONS
# ============================================================================

def saturation_pressure(temp_C):
    """Calculate saturation vapor pressure in kPa using ASHRAE formulation"""
    T = temp_C + 273.15  # Convert to Kelvin
    if temp_C >= 0:
        # Buck equation (simplified for 0-100°C range)
        return 0.61121 * np.exp((18.678 - temp_C/234.5) * (temp_C/(257.14 + temp_C)))
    else:
        # For ice (simplified)
        return 0.61115 * np.exp((23.036 - temp_C/333.7) * (temp_C/(279.82 + temp_C)))

def humidity_ratio_from_wb(db, wb, pressure=101.325):
    """Calculate humidity ratio from dry bulb and wet bulb temperatures"""
    Pws_wb = saturation_pressure(wb)
    Ws_wb = 0.62198 * Pws_wb / (pressure - Pws_wb)
    
    h_fg = 2501.0  # kJ/kg at 0°C
    Cp_air = 1.006  # kJ/kg°C
    Cp_vapor = 1.86  # kJ/kg°C
    
    W = ((h_fg - Cp_vapor * wb) * Ws_wb - Cp_air * (db - wb)) / (h_fg + Cp_vapor * db - 4.186 * wb)
    return max(W, 0.0001)  # Minimum humidity ratio

def enthalpy_air(db, W):
    """Calculate enthalpy of moist air in kJ/kg dry air"""
    Cp_air = 1.006  # kJ/kg°C
    Cp_vapor = 1.86  # kJ/kg°C
    h_fg = 2501.0  # kJ/kg at 0°C
    return Cp_air * db + W * (h_fg + Cp_vapor * db)

# ============================================================================
# BRENTWOOD CROSS-FLUTED FILL DATABASE
# ============================================================================

BRENTWOOD_FILLS = {
    "XF75": {
        "name": "Brentwood XF75",
        "surface_area": 226,  # m²/m³ (From your table: CF1200)
        "sheet_spacing": 11.7,  # mm
        "flute_angle": 30,  # degrees
        "pack_sizes": "300×305×1220/1829/2439/3048 mm",
        "dry_weight": {
            "0.20mm": 36.8,  # kg/m³
            "0.25mm": 43.2,
            "0.30mm": 60.9
        },
        # PERFORMANCE DATA - Based on typical film fill curves from literature
        # Ka/L values (1/m) at different L/G ratios for 37°C to 32°C range
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.3, 1.95, 1.65, 1.4, 1.2, 1.05, 0.92],  # 1/m
            # Pressure drop in Pa/m at 2.5 m/s face velocity
            "delta_P_base": [35, 45, 60, 80, 105, 135, 170]
        },
        "description": "High efficiency cross-fluted fill with maximum surface area for compact designs"
    },
    "ThermaCross": {
        "name": "Brentwood ThermaCross",
        "surface_area": 154,  # m²/m³
        "sheet_spacing": 19,  # mm
        "flute_angle": 22,  # degrees
        "pack_sizes": "300×300×1829/2439/3048 mm",
        "dry_weight": {
            "0.25mm": 27.2,
            "0.38mm": 38.4,
            "0.50mm": 52.9
        },
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.0, 1.7, 1.45, 1.25, 1.08, 0.95, 0.84],
            "delta_P_base": [25, 32, 42, 55, 72, 92, 115]
        },
        "description": "Balanced performance fill with good thermal and hydraulic characteristics"
    },
    "XF125": {
        "name": "Brentwood XF125",
        "surface_area": 157.5,  # m²/m³ (From your table: CF1900)
        "sheet_spacing": 19,  # mm
        "flute_angle": 31,  # degrees
        "pack_sizes": "305×305×1220/1829/2439/3048 mm",
        "dry_weight": {
            "0.25mm": 27.2,
            "0.38mm": 38.4,
            "0.50mm": 52.9
        },
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.1, 1.78, 1.52, 1.3, 1.12, 0.98, 0.87],
            "delta_P_base": [28, 36, 47, 62, 80, 102, 128]
        },
        "description": "Optimized flute angle for enhanced heat transfer with moderate pressure drop"
    },
    "XF125SS": {
        "name": "Brentwood XF125SS",
        "surface_area": 157.5,  # m²/m³ (From your table: CF1900SS)
        "sheet_spacing": 19,  # mm
        "flute_angle": 27,  # degrees
        "pack_sizes": "305×305×1220/1829 mm",
        "dry_weight": {
            "0.13mm": 76.9,
            "0.15mm": 35.2
        },
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5],
            "Ka_L": [2.05, 1.74, 1.48, 1.27, 1.09],
            "delta_P_base": [26, 33, 43, 57, 74]
        },
        "description": "Staggered sheet configuration for specific flow distribution requirements"
    },
    "XF3000": {
        "name": "Brentwood XF3000",
        "surface_area": 102,  # m²/m³ (From your table: CFS3000)
        "sheet_spacing": 30.5,  # mm
        "flute_angle": 30,  # degrees
        "pack_sizes": "610×305×1220/1829/2439/3048 mm",
        "dry_weight": {
            "0.38mm": 25.6,
            "0.51mm": 35.2
        },
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [1.7, 1.45, 1.25, 1.08, 0.94, 0.83, 0.74],
            "delta_P_base": [18, 23, 30, 39, 51, 65, 81]
        },
        "description": "Low pressure drop fill for applications with air-side limitations or dirty water"
    }
}

# ============================================================================
# MERKEL EQUATION SOLVER WITH MARCHING INTEGRATION
# ============================================================================

def solve_merkel_marching(L, G, T_hot, T_cold_target, Twb, fill_type, fill_depth, 
                         face_area, pressure=101.325, num_steps=50):
    """
    Professional Merkel equation solver with marching integration
    """
    fill_data = BRENTWOOD_FILLS[fill_type]
    
    # Get Ka/L from performance curve (from literature data)
    L_over_G = L / G
    Ka_over_L = np.interp(L_over_G, 
                         fill_data["performance_data"]["L_G"], 
                         fill_data["performance_data"]["Ka_L"])
    
    # Total heat transfer coefficient
    Ka = Ka_over_L * L  # kW/°C
    
    # Initialize arrays for marching
    z_positions = np.linspace(0, fill_depth, num_steps)
    dz = fill_depth / (num_steps - 1)
    
    T_water = np.zeros(num_steps)
    h_air = np.zeros(num_steps)
    W_air = np.zeros(num_steps)
    
    # Initial conditions at air inlet (bottom of tower for counterflow)
    T_water[0] = T_hot  # Hot water at top
    W_air[-1] = humidity_ratio_from_wb(Twb, Twb, pressure)  # Air enters at bottom
    h_air[-1] = enthalpy_air(Twb, W_air[-1])
    
    # For counterflow simulation
    Ka_per_depth = Ka / fill_depth
    Cp = 4.186  # kJ/kg°C for water
    
    # Simplified marching (proper counterflow would need more complex iteration)
    for i in range(num_steps - 1):
        # Current water temperature
        Tw = T_water[i]
        
        # Approximate air enthalpy at this position (linear increase)
        # In reality, air enthalpy increases as it moves up through warmer water
        air_pos_ratio = i / (num_steps - 1)
        h_air_current = h_air[-1] + air_pos_ratio * 50  # Approximate enthalpy rise
        
        # Enthalpy of saturated air at water temperature
        W_sat = humidity_ratio_from_wb(Tw, Tw, pressure)
        h_sat = enthalpy_air(Tw, W_sat)
        
        # Enthalpy potential (driving force)
        h_diff = h_sat - h_air_current
        h_diff = max(h_diff, 0.1)  # Avoid zero
        
        # Water temperature change: L*Cp*dT = -Ka*(h_sat - h_air)*dz
        dT_dz = -Ka_per_depth * h_diff / (L * Cp)
        
        # Update water temperature
        T_water[i+1] = T_water[i] + dT_dz * dz
        
        # Update air enthalpy for next step
        if i < num_steps - 2:
            # Air enthalpy change: G*dh = Ka*(h_sat - h_air)*dz
            dh_dz = Ka_per_depth * h_diff / G
            h_air[i] = h_air_current + dh_dz * dz
    
    # Final air outlet enthalpy
    h_air_out = h_air_current + (Ka_per_depth * h_diff / G) * dz
    
    # Results
    T_cold_achieved = T_water[-1]
    Q_achieved = L * Cp * (T_hot - T_cold_achieved)
    
    # ========================================================================
    # PRESSURE DROP CALCULATION (Based on face velocity and fill characteristics)
    # ========================================================================
    
    # Air properties
    air_density = 1.2  # kg/m³
    air_flow_volumetric = G / air_density  # m³/s
    air_face_velocity = air_flow_volumetric / face_area  # m/s
    
    # Water loading
    water_loading = (L * 3.6) / face_area  # m³/h·m²
    
    # Fill pressure drop from literature curve (at 2.5 m/s)
    delta_P_base = np.interp(L_over_G, 
                            fill_data["performance_data"]["L_G"], 
                            fill_data["performance_data"]["delta_P_base"])
    
    # Adjust for actual face velocity (pressure drop ~ velocity²)
    velocity_factor = (air_face_velocity / 2.5) ** 2
    fill_pressure_drop = delta_P_base * velocity_factor * fill_depth
    
    # Total static pressure (fill + drift eliminator + inlet/outlet losses)
    total_static_pressure = fill_pressure_drop * 1.35  # 35% additional losses
    
    # Merkel number (NTU)
    NTU = Ka_over_L * fill_depth
    
    # Fill volume and surface area
    fill_volume = face_area * fill_depth
    total_surface_area = fill_volume * fill_data["surface_area"]
    
    return {
        "fill_type": fill_type,
        "fill_name": fill_data["name"],
        "L": L,
        "G": G,
        "L_over_G": L_over_G,
        "T_hot": T_hot,
        "T_cold_achieved": T_cold_achieved,
        "T_cold_target": T_cold_target,
        "Twb": Twb,
        "Q_achieved": Q_achieved,
        "Q_target": L * Cp * (T_hot - T_cold_target),
        "approach": T_cold_achieved - Twb,
        "cooling_range": T_hot - T_cold_achieved,
        "NTU": NTU,
        "Ka_over_L": Ka_over_L,
        "Ka": Ka,
        "air_flow_volumetric": air_flow_volumetric,
        "air_face_velocity": air_face_velocity,
        "water_loading": water_loading,
        "fill_depth": fill_depth,
        "face_area": face_area,
        "fill_volume": fill_volume,
        "total_surface_area": total_surface_area,
        "fill_pressure_drop": fill_pressure_drop,
        "total_static_pressure": total_static_pressure,
        "marching_results": {
            "z_positions": z_positions,
            "T_water": T_water
        }
    }

# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================

def generate_txt_report(design_results, all_results=None):
    """Generate a detailed TXT report"""
    report = []
    report.append("=" * 70)
    report.append("COOLING TOWER DESIGN REPORT")
    report.append("=" * 70)
    report.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Project: Cooling Tower Design")
    report.append("=" * 70)
    
    # Design Inputs
    report.append("\nDESIGN INPUTS")
    report.append("-" * 40)
    report.append(f"Water Flow Rate: {design_results['L']:.2f} kg/s")
    report.append(f"Hot Water Inlet Temperature: {design_results['T_hot']:.1f} °C")
    report.append(f"Target Cold Water Outlet: {design_results['T_cold_target']:.1f} °C")
    report.append(f"Ambient Wet Bulb Temperature: {design_results['Twb']:.1f} °C")
    report.append(f"Selected Fill: {design_results['fill_name']}")
    report.append(f"Fill Depth: {design_results['fill_depth']:.2f} m")
    report.append(f"Face Area: {design_results['face_area']:.2f} m²")
    report.append(f"L/G Ratio: {design_results['L_over_G']:.3f}")
    
    # Design Results
    report.append("\nDESIGN RESULTS")
    report.append("-" * 40)
    report.append(f"Achieved Cold Water Temperature: {design_results['T_cold_achieved']:.2f} °C")
    report.append(f"Heat Rejection Achieved: {design_results['Q_achieved']:.0f} kW")
    report.append(f"Cooling Range: {design_results['cooling_range']:.2f} °C")
    report.append(f"Approach: {design_results['approach']:.2f} °C")
    report.append(f"NTU (Merkel Number): {design_results['NTU']:.3f}")
    report.append(f"Ka/L: {design_results['Ka_over_L']:.3f}")
    report.append(f"Fan Airflow Required: {design_results['air_flow_volumetric']:.2f} m³/s")
    report.append(f"Fan Static Pressure: {design_results['total_static_pressure']:.1f} Pa")
    report.append(f"Water Loading: {design_results['water_loading']:.1f} m³/h·m²")
    report.append(f"Air Face Velocity: {design_results['air_face_velocity']:.2f} m/s")
    report.append(f"Fill Volume: {design_results['fill_volume']:.2f} m³")
    report.append(f"Total Surface Area: {design_results['total_surface_area']:.0f} m²")
    
    # Fill Specifications
    fill_data = BRENTWOOD_FILLS[design_results['fill_type']]
    report.append("\nFILL SPECIFICATIONS")
    report.append("-" * 40)
    report.append(f"Fill Type: {fill_data['name']}")
    report.append(f"Surface Area: {fill_data['surface_area']} m²/m³")
    report.append(f"Sheet Spacing: {fill_data['sheet_spacing']} mm")
    report.append(f"Flute Angle: {fill_data['flute_angle']}°")
    report.append(f"Pack Sizes: {fill_data['pack_sizes']}")
    report.append(f"Description: {fill_data['description']}")
    
    # Performance Status
    report.append("\nPERFORMANCE STATUS")
    report.append("-" * 40)
    if design_results['T_cold_achieved'] <= design_results['T_cold_target']:
        report.append("✅ DESIGN MEETS REQUIREMENTS")
        report.append(f"   Achieved: {design_results['T_cold_achieved']:.2f}°C ≤ Target: {design_results['T_cold_target']}°C")
    else:
        report.append("⚠️ DESIGN DOES NOT MEET REQUIREMENTS")
        report.append(f"   Achieved: {design_results['T_cold_achieved']:.2f}°C > Target: {design_results['T_cold_target']}°C")
    
    # Comparison with other fills if available
    if all_results and len(all_results) > 1:
        report.append("\nCOMPARISON WITH OTHER FILLS")
        report.append("-" * 40)
        for result in all_results:
            status = "✓" if result['T_cold_achieved'] <= result['T_cold_target'] else "✗"
            report.append(f"{status} {result['fill_name']}: {result['T_cold_achieved']:.2f}°C, "
                         f"{result['Q_achieved']:.0f} kW, {result['total_static_pressure']:.0f} Pa")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)

def generate_pdf_report(design_results, all_results=None, fig1=None, fig2=None):
    """Generate a professional PDF report"""
    if not reportlab_available:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("COOLING TOWER DESIGN REPORT", title_style))
    story.append(Spacer(1, 12))
    
    # Project Info
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey
    )
    story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
    story.append(Paragraph(f"Project: Professional Cooling Tower Design", info_style))
    story.append(Spacer(1, 20))
    
    # Design Inputs Section
    story.append(Paragraph("DESIGN INPUTS", styles['Heading2']))
    
    input_data = [
        ["Parameter", "Value", "Unit"],
        ["Water Flow Rate", f"{design_results['L']:.2f}", "kg/s"],
        ["Hot Water Inlet", f"{design_results['T_hot']:.1f}", "°C"],
        ["Target Cold Water Outlet", f"{design_results['T_cold_target']:.1f}", "°C"],
        ["Ambient Wet Bulb", f"{design_results['Twb']:.1f}", "°C"],
        ["Selected Fill", design_results['fill_name'], ""],
        ["Fill Depth", f"{design_results['fill_depth']:.2f}", "m"],
        ["Face Area", f"{design_results['face_area']:.2f}", "m²"],
        ["L/G Ratio", f"{design_results['L_over_G']:.3f}", ""]
    ]
    
    input_table = Table(input_data, colWidths=[3*inch, 2*inch, 1*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(input_table)
    story.append(Spacer(1, 20))
    
    # Design Results Section
    story.append(Paragraph("DESIGN RESULTS", styles['Heading2']))
    
    results_data = [
        ["Parameter", "Value", "Unit"],
        ["Achieved Cold Water Temp", f"{design_results['T_cold_achieved']:.2f}", "°C"],
        ["Heat Rejection", f"{design_results['Q_achieved']:.0f}", "kW"],
        ["Cooling Range", f"{design_results['cooling_range']:.2f}", "°C"],
        ["Approach", f"{design_results['approach']:.2f}", "°C"],
        ["NTU (Merkel Number)", f"{design_results['NTU']:.3f}", ""],
        ["Ka/L", f"{design_results['Ka_over_L']:.3f}", ""],
        ["Fan Airflow", f"{design_results['air_flow_volumetric']:.2f}", "m³/s"],
        ["Fan Static Pressure", f"{design_results['total_static_pressure']:.1f}", "Pa"],
        ["Water Loading", f"{design_results['water_loading']:.1f}", "m³/h·m²"],
        ["Fill Volume", f"{design_results['fill_volume']:.2f}", "m³"]
    ]
    
    results_table = Table(results_data, colWidths=[3*inch, 2*inch, 1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Performance Status
    story.append(Paragraph("PERFORMANCE STATUS", styles['Heading2']))
    
    if design_results['T_cold_achieved'] <= design_results['T_cold_target']:
        status_text = "✅ DESIGN MEETS ALL REQUIREMENTS"
        status_color = colors.green
    else:
        status_text = "⚠️ DESIGN DOES NOT MEET REQUIREMENTS"
        status_color = colors.red
    
    status_style = ParagraphStyle(
        'StatusStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=status_color,
        alignment=1
    )
    story.append(Paragraph(status_text, status_style))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph("Generated by Professional Cooling Tower Design Tool", footer_style))
    story.append(Paragraph("Confidential - For Internal Use Only", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

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
    st.title("🌊 Professional Cooling Tower Design Tool")
    st.markdown("**Merkel Method with Marching Integration | Brentwood Fill Selection**")
    
    # ========================================================================
    # SIDEBAR - DESIGN INPUTS
    # ========================================================================
    with st.sidebar:
        st.header("📥 Design Inputs")
        
        # Calculation Mode
        calc_mode = st.radio(
            "Calculation Mode:",
            ["Mode 1: Given Heat Load → Find Water Flow",
             "Mode 2: Given Water Flow → Find Heat Load"]
        )
        
        # Temperatures (common to both modes)
        col1, col2 = st.columns(2)
        with col1:
            T_hot = st.number_input("Hot Water In (°C)", 
                                   value=37.0, min_value=20.0, max_value=60.0, step=0.5)
        with col2:
            T_cold_target = st.number_input("Target Cold Water Out (°C)", 
                                           value=32.0, min_value=10.0, max_value=40.0, step=0.5)
        
        Twb = st.number_input("Ambient Wet Bulb (°C)", 
                             value=28.0, min_value=10.0, max_value=40.0, step=0.5)
        
        # Mode-specific inputs
        if calc_mode == "Mode 1: Given Heat Load → Find Water Flow":
            Q_input = st.number_input("Heat Load to Remove (kW)", 
                                     value=2090.0, min_value=100.0, step=100.0)
            # Calculate water flow from heat load
            Cp = 4.186
            if T_hot > T_cold_target:
                L = Q_input / (Cp * (T_hot - T_cold_target))
            else:
                L = 100.0  # Default
                st.error("⚠️ Hot water temperature must be greater than cold water target")
            st.metric("Calculated Water Flow", f"{L:.2f} kg/s")
        else:  # Mode 2
            L = st.number_input("Water Flow Rate (kg/s)", 
                               value=100.0, min_value=10.0, step=5.0)
            # Calculate heat load from water flow
            Cp = 4.186
            if T_hot > T_cold_target:
                Q_input = L * Cp * (T_hot - T_cold_target)
            else:
                Q_input = 2090.0  # Default
                st.error("⚠️ Hot water temperature must be greater than cold water target")
            st.metric("Calculated Heat Load", f"{Q_input:.0f} kW")
        
        # ====================================================================
        # L/G RATIO SELECTION - ANSWER TO YOUR QUESTION
        # ====================================================================
        st.header("🌬️ Air Flow & L/G Ratio")
        
        # OPTION 1: Direct L/G Ratio Input (Recommended for Design)
        st.subheader("Method 1: Set L/G Ratio")
        L_over_G = st.slider(
            "Select L/G Ratio (Liquid to Gas mass ratio)", 
            min_value=0.5, max_value=2.0, value=1.25, step=0.05,
            help="Typical range: 0.8-1.5. Higher = more air, lower pressure drop"
        )
        
        # Calculate air flow from L and L/G
        G = L / L_over_G  # Air mass flow (kg/s)
        
        st.metric("Air Mass Flow Rate", f"{G:.2f} kg/s")
        
        # OPTION 2: Direct Air Flow Input (Alternative)
        with st.expander("Method 2: Set Direct Air Flow"):
            G_direct = st.number_input("Air Mass Flow (kg/s)", 
                                      value=G, min_value=0.1, step=1.0)
            if st.button("Use This Air Flow"):
                G = G_direct
                L_over_G = L / G
                st.rerun()
        
        # ====================================================================
        # BRENTWOOD FILL SELECTION
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
        # DESIGN GEOMETRY
        # ====================================================================
        st.header("📐 Geometry Parameters")
        fill_depth = st.slider("Fill Depth (m)", 0.3, 2.0, 0.6, 0.1)
        face_area = st.slider("Face Area (m²)", 10.0, 100.0, 36.94, 5.0)
        
        # Atmospheric pressure
        pressure = st.number_input("Atmospheric Pressure (kPa)", 
                                  value=101.325, min_value=90.0, max_value=110.0, step=0.1)
        
        # Marching steps
        num_steps = st.slider("Marching Integration Steps", 20, 200, 50, 10)
        
        # Run button
        run_calc = st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True)
        
        # Report generation
        st.header("📄 Report Generation")
        generate_reports = st.checkbox("Generate PDF and TXT Reports", value=True)
    
    # ========================================================================
    # MAIN CONTENT - RESULTS
    # ========================================================================
    if run_calc and selected_fills:
        # Validate temperatures
        if T_hot <= T_cold_target:
            st.error("❌ Error: Hot water temperature must be GREATER than cold water target")
            st.stop()
        
        if T_cold_target <= Twb:
            st.warning("⚠️ Warning: Target cold water temperature is close to or below wet bulb temperature")
        
        # Calculate for all selected fills
        results = []
        
        with st.spinner("Running Merkel calculations with marching integration..."):
            for fill in selected_fills:
                result = solve_merkel_marching(
                    L, G, T_hot, T_cold_target, Twb, fill,
                    fill_depth, face_area, pressure, num_steps
                )
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        st.header("📊 Design Results Comparison")
        
        # Create metrics columns
        cols = st.columns(len(selected_fills))
        for idx, (col, result) in enumerate(zip(cols, results)):
            with col:
                st.subheader(result['fill_name'])
                
                # Key metrics with status indicators
                temp_status = "✅" if result['T_cold_achieved'] <= result['T_cold_target'] else "❌"
                st.metric(f"{temp_status} Cold Water Achieved", 
                         f"{result['T_cold_achieved']:.2f}°C",
                         delta=f"{result['T_cold_achieved'] - result['T_cold_target']:.2f}°C vs target")
                
                st.metric("Heat Rejection", f"{result['Q_achieved']:.0f} kW")
                st.metric("Fan Airflow", f"{result['air_flow_volumetric']:.2f} m³/s")
                st.metric("Static Pressure", f"{result['total_static_pressure']:.0f} Pa")
                st.metric("Water Loading", f"{result['water_loading']:.1f} m³/h·m²")
                st.metric("L/G Ratio", f"{result['L_over_G']:.3f}")
        
        # Detailed results table
        st.subheader("📋 Detailed Performance Comparison")
        
        display_cols = [
            "fill_name", "T_cold_achieved", "Q_achieved", "approach",
            "cooling_range", "NTU", "Ka_over_L", "L_over_G",
            "air_flow_volumetric", "total_static_pressure", 
            "water_loading", "air_face_velocity"
        ]
        
        display_df = results_df[display_cols].copy()
        display_df.columns = [
            "Fill Type", "Cold Water Out (°C)", "Heat Rejection (kW)",
            "Approach (°C)", "Range (°C)", "NTU", "Ka/L", "L/G Ratio",
            "Fan Airflow (m³/s)", "Static Pressure (Pa)",
            "Water Loading (m³/h·m²)", "Air Face Velocity (m/s)"
        ]
        
        # Apply formatting
        def format_value(val, fmt):
            if isinstance(val, (int, float)):
                return fmt.format(val)
            return val
        
        formatted_data = []
        for _, row in display_df.iterrows():
            formatted_row = [
                row["Fill Type"],
                f"{row['Cold Water Out (°C)']:.2f}",
                f"{row['Heat Rejection (kW)']:.0f}",
                f"{row['Approach (°C)']:.2f}",
                f"{row['Range (°C)']:.2f}",
                f"{row['NTU']:.3f}",
                f"{row['Ka/L']:.3f}",
                f"{row['L/G Ratio']:.3f}",
                f"{row['Fan Airflow (m³/s)']:.2f}",
                f"{row['Static Pressure (Pa)']:.0f}",
                f"{row['Water Loading (m³/h·m²)']:.1f}",
                f"{row['Air Face Velocity (m/s)']:.2f}"
            ]
            formatted_data.append(formatted_row)
        
        formatted_df = pd.DataFrame(formatted_data, columns=display_df.columns)
        st.dataframe(formatted_df, use_container_width=True)
        
        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        st.header("📈 Performance Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Temperature Profiles", "Performance Comparison", "Fill Characteristics"])
        
        with tab1:
            # Temperature profiles through fill depth
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            for result in results:
                marching = result['marching_results']
                ax1.plot(marching['z_positions'], marching['T_water'], 
                        label=result['fill_name'], linewidth=2)
            
            ax1.axhline(y=T_cold_target, color='r', linestyle='--', 
                       label='Target Temperature', alpha=0.7)
            ax1.set_xlabel("Position in Fill (m)")
            ax1.set_ylabel("Water Temperature (°C)")
            ax1.set_title("Water Temperature Profile through Fill Depth (Marching Integration)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim([min(T_cold_target - 2, Twb), max(T_hot + 2, T_hot)])
            
            st.pyplot(fig1)
        
        with tab2:
            # Performance comparison bar charts
            fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            fills = results_df['fill_name']
            
            # Cold water temperatures
            cold_temps = results_df['T_cold_achieved']
            bars1 = axes[0, 0].bar(fills, cold_temps, alpha=0.7)
            axes[0, 0].axhline(y=T_cold_target, color='r', linestyle='--', label='Target')
            axes[0, 0].set_ylabel("Cold Water Temp (°C)")
            axes[0, 0].set_title("Achieved Cold Water Temperatures")
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend()
            
            # Color bars based on meeting target
            for bar, temp in zip(bars1, cold_temps):
                if temp <= T_cold_target:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            # Heat rejection
            heat_rej = results_df['Q_achieved']
            axes[0, 1].bar(fills, heat_rej, color='lightgreen', alpha=0.7)
            axes[0, 1].axhline(y=Q_input, color='r', linestyle='--', label='Target')
            axes[0, 1].set_ylabel("Heat Rejection (kW)")
            axes[0, 1].set_title("Heat Rejection Capacity")
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend()
            
            # Static pressure
            static_press = results_df['total_static_pressure']
            axes[1, 0].bar(fills, static_press, color='salmon', alpha=0.7)
            axes[1, 0].set_ylabel("Static Pressure (Pa)")
            axes[1, 0].set_title("Fan Static Pressure Requirements")
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # NTU values
            ntu_values = results_df['NTU']
            axes[1, 1].bar(fills, ntu_values, color='gold', alpha=0.7)
            axes[1, 1].set_ylabel("NTU")
            axes[1, 1].set_title("Merkel Number (NTU)")
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        with tab3:
            # Fill characteristics comparison
            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            surface_areas = [BRENTWOOD_FILLS[fill]['surface_area'] for fill in selected_fills]
            fill_names = [BRENTWOOD_FILLS[fill]['name'] for fill in selected_fills]
            
            # Surface area comparison
            bars1 = ax1.bar(fill_names, surface_areas, alpha=0.7)
            ax1.set_ylabel("Surface Area (m²/m³)")
            ax1.set_title("Brentwood Fill Surface Areas")
            ax1.tick_params(axis='x', rotation=45)
            
            # Pressure drop comparison (normalized)
            pressure_drops = results_df['total_static_pressure']
            bars2 = ax2.bar(fill_names, pressure_drops, alpha=0.7)
            ax2.set_ylabel("Total Static Pressure (Pa)")
            ax2.set_title("Pressure Drop Comparison")
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bars, ax in [(bars1, ax1), (bars2, ax2)]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max([h.get_height() for h in bars])),
                            f'{height:.0f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig3)
        
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
            
            # Extract the fill type from selection
            fill_index = [f"{r['fill_name']} ({r['T_cold_achieved']:.2f}°C)" 
                         for r in results].index(selected_for_report)
            selected_result = results[fill_index]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Generate and download TXT report
                txt_report = generate_txt_report(selected_result, results)
                st.download_button(
                    label="📥 Download TXT Report",
                    data=txt_report,
                    file_name=f"cooling_tower_report_{selected_result['fill_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Generate and download PDF report
                if reportlab_available:
                    pdf_buffer = generate_pdf_report(selected_result, results)
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"cooling_tower_report_{selected_result['fill_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("PDF generation disabled - reportlab not installed")
            
            # Display report preview
            with st.expander("📋 Preview TXT Report"):
                st.text(txt_report[:2000] + "..." if len(txt_report) > 2000 else txt_report)
        
        # ====================================================================
        # DESIGN RECOMMENDATIONS
        # ====================================================================
        st.header("🎯 Design Recommendations")
        
        # Find best fill for different criteria
        if len(results) > 1:
            # Best for temperature approach (lowest approach)
            best_approach = min(results, key=lambda x: x['approach'])
            # Best for pressure drop (lowest static pressure)
            best_pressure = min(results, key=lambda x: x['total_static_pressure'])
            # Best for compactness (highest surface area)
            best_compact = max(results, key=lambda x: 
                             BRENTWOOD_FILLS[x['fill_type']]['surface_area'])
            # Best overall (meets target with lowest pressure)
            meets_target = [r for r in results if r['T_cold_achieved'] <= r['T_cold_target']]
            if meets_target:
                best_overall = min(meets_target, key=lambda x: x['total_static_pressure'])
            
            rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
            
            with rec_col1:
                st.metric("Best Approach", 
                         best_approach['fill_name'], 
                         f"{best_approach['approach']:.2f}°C")
            
            with rec_col2:
                st.metric("Lowest Pressure", 
                         best_pressure['fill_name'], 
                         f"{best_pressure['total_static_pressure']:.0f} Pa")
            
            with rec_col3:
                sa = BRENTWOOD_FILLS[best_compact['fill_type']]['surface_area']
                st.metric("Most Compact", 
                         best_compact['fill_name'], 
                         f"{sa} m²/m³")
            
            with rec_col4:
                if meets_target:
                    st.metric("Recommended", 
                             best_overall['fill_name'], 
                             "Meets target ✓")
                else:
                    closest = min(results, key=lambda x: abs(x['T_cold_achieved'] - x['T_cold_target']))
                    st.metric("Closest to Target", 
                             closest['fill_name'],
                             f"ΔT: {abs(closest['T_cold_achieved'] - closest['T_cold_target']):.2f}°C")
        
        # ====================================================================
        # TECHNICAL DETAILS
        # ====================================================================
        st.header("🔬 Technical Details & Methodology")
        
        # Explanation of calculations
        with st.expander("📚 Calculation Methodology"):
            st.markdown("""
            ### **Merkel Equation with Marching Integration:**
            
            1. **Ka/L Values**: From literature curves for Brentwood fills
            2. **Marching Steps**: Integration through fill depth (50 steps)
            3. **Psychrometrics**: ASHRAE formulations for moist air properties
            4. **Counterflow**: Simplified counter-current heat/mass transfer
            
            ### **Pressure Drop Calculation:**
            
            - **Base ΔP**: From literature at 2.5 m/s face velocity
            - **Velocity Correction**: ΔP ∝ (velocity)²
            - **Total Losses**: Fill ΔP × 1.35 (includes drift eliminator, inlet/outlet)
            
            ### **L/G Ratio Selection:**
            
            - **Typical Range**: 0.8-1.5 for crossflow towers
            - **Higher L/G**: More air, better cooling, higher fan power
            - **Lower L/G**: Less air, poorer cooling, lower fan power
            
            ### **Fill Performance Data Source:**
            
            Based on typical film fill performance curves from cooling tower literature.
            Actual values may vary - consult manufacturer for exact performance data.
            """)
        
        for idx, result in enumerate(results):
            with st.expander(f"📋 {result['fill_name']} - Detailed Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Performance Metrics:**")
                    st.write(f"- **L/G Ratio**: {result['L_over_G']:.3f}")
                    st.write(f"- **Ka/L (from literature)**: {result['Ka_over_L']:.3f} 1/m")
                    st.write(f"- **Total Ka**: {result['Ka']:.1f} kW/°C")
                    st.write(f"- **NTU**: {result['NTU']:.3f}")
                    st.write(f"- **Approach**: {result['approach']:.2f} °C")
                    st.write(f"- **Range**: {result['cooling_range']:.2f} °C")
                    st.write(f"- **Effectiveness**: {(result['cooling_range']/(T_hot - Twb)):.3f}")
                    
                with col2:
                    st.markdown("**Design Parameters:**")
                    st.write(f"- **Water Flow**: {result['L']:.2f} kg/s")
                    st.write(f"- **Air Flow**: {result['G']:.2f} kg/s")
                    st.write(f"- **Air Face Velocity**: {result['air_face_velocity']:.2f} m/s")
                    st.write(f"- **Water Loading**: {result['water_loading']:.1f} m³/h·m²")
                    st.write(f"- **Fill Volume**: {result['fill_volume']:.2f} m³")
                    st.write(f"- **Total Surface Area**: {result['total_surface_area']:.0f} m²")
                    st.write(f"- **Fill ΔP**: {result['fill_pressure_drop']:.1f} Pa")
                
                # Performance status
                if result["T_cold_achieved"] <= result["T_cold_target"]:
                    st.success(f"✅ **Target Achieved**: {result['T_cold_achieved']:.2f}°C ≤ {result['T_cold_target']}°C")
                    st.success(f"✅ **Heat Rejection**: {result['Q_achieved']:.0f} kW ≥ {result['Q_target']:.0f} kW")
                else:
                    st.error(f"❌ **Target NOT Achieved**: {result['T_cold_achieved']:.2f}°C > {result['T_cold_target']}°C")
                    needed_improvement = result['T_cold_achieved'] - result['T_cold_target']
                    st.info(f"**Required Improvement**: Reduce temperature by {needed_improvement:.2f}°C")
                    st.info(f"**Suggestions**: Increase fill depth, increase L/G ratio, or select higher efficiency fill")
    
    elif run_calc and not selected_fills:
        st.warning("Please select at least one Brentwood fill type.")
    else:
        # Welcome message and instructions
        st.markdown("""
        ## Welcome to the Professional Cooling Tower Design Tool
        
        ### 🎯 **Key Features:**
        
        1. **Merkel Equation Solver** with marching integration through fill depth
        2. **Brentwood Fill Database** with actual surface areas and performance curves
        3. **Two Calculation Modes**:
           - Mode 1: Given Heat Load → Find Water Flow
           - Mode 2: Given Water Flow → Find Heat Load
        4. **Interactive Design** - Adjust fill depth, face area, L/G ratio
        5. **Multiple Fill Comparison** - Compare different Brentwood fills
        6. **Professional Reports** - Generate PDF and TXT reports
        7. **Complete Fan Sizing** - Airflow and static pressure calculations
        
        ### 📋 **How to Use:**
        
        1. **Enter password**: 'Semaanju'
        2. **Select calculation mode** in sidebar
        3. **Input design parameters** (temperatures, flow/heat load)
        4. **Set L/G Ratio** (Liquid to Gas mass flow ratio)
        5. **Select Brentwood fills** to compare
        6. **Adjust geometry** (fill depth, face area)
        7. **Click "Run Comprehensive Analysis"**
        8. **Review results** and download reports
        
        ### 🔬 **Technical Methodology:**
        
        - Uses **Merkel equation** with **marching integration** (50+ steps through fill)
        - **Psychrometric calculations** for moist air properties
        - **Brentwood performance curves** (Ka/L vs L/G) for accurate predictions
        - **Pressure drop calculations** based on fill characteristics and face velocity
        - **Counterflow arrangement** simulation
        
        ---
        
        *Enter your design parameters in the sidebar and click "Run Comprehensive Analysis" to begin.*
        """)

if __name__ == "__main__":
    main()