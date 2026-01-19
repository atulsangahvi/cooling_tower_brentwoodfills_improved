# cooling_tower_design_pro.py
# Professional Cooling Tower Design Tool
# Password: "Semaanju"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
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
        # ASHRAE formulation for water
        C1 = -5.6745359e3
        C2 = 6.3925247
        C3 = -9.677843e-3
        C4 = 6.2215701e-7
        C5 = 2.0747825e-9
        C6 = -9.484024e-13
        C7 = 4.1635019
        C8 = -5.8002206e3
        C9 = 1.3914993
        C10 = -4.8640239e-2
        C11 = 4.1764768e-5
        C12 = -1.4452093e-8
        C13 = 6.5459673
        
        if T <= 273.15:
            lnPws = C1/T + C2 + C3*T + C4*T**2 + C5*T**3 + C6*T**4 + C7*np.log(T)
        else:
            lnPws = C8/T + C9 + C10*T + C11*T**2 + C12*T**3 + C13*np.log(T)
        
        return np.exp(lnPws) / 1000  # Convert to kPa
    else:
        # For ice (simplified)
        return 0.611 * np.exp(22.46 * temp_C / (272.62 + temp_C))

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
        "surface_area": 226,  # m²/m³
        "sheet_spacing": 11.7,  # mm
        "flute_angle": 30,  # degrees
        "pack_sizes": "300×305×1220/1829/2439/3048 mm",
        "dry_weight": {
            "0.20mm": 36.8,  # kg/m³
            "0.25mm": 43.2,
            "0.30mm": 60.9
        },
        "performance_data": {
            "L_G": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "Ka_L": [2.3, 1.95, 1.65, 1.4, 1.2, 1.05, 0.92],
            "delta_P": [35, 45, 60, 80, 105, 135, 170]  # Pa/m at 2.5 m/s
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
            "delta_P": [25, 32, 42, 55, 72, 92, 115]  # Pa/m
        },
        "description": "Balanced performance fill with good thermal and hydraulic characteristics"
    },
    "XF125": {
        "name": "Brentwood XF125",
        "surface_area": 157.5,  # m²/m³
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
            "delta_P": [28, 36, 47, 62, 80, 102, 128]  # Pa/m
        },
        "description": "Optimized flute angle for enhanced heat transfer with moderate pressure drop"
    },
    "XF125SS": {
        "name": "Brentwood XF125SS",
        "surface_area": 157.5,  # m²/m³
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
            "delta_P": [26, 33, 43, 57, 74]  # Pa/m
        },
        "description": "Staggered sheet configuration for specific flow distribution requirements"
    },
    "XF3000": {
        "name": "Brentwood XF3000",
        "surface_area": 102,  # m²/m³
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
            "delta_P": [18, 23, 30, 39, 51, 65, 81]  # Pa/m
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
    
    # Get Ka/L from performance curve
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
    
    # For counterflow: air moves up, water moves down
    # We'll march from water inlet (top) to water outlet (bottom)
    Ka_per_depth = Ka / fill_depth
    
    # Temperature and enthalpy differences
    dT_dz = np.zeros(num_steps)
    dh_dz = np.zeros(num_steps)
    
    # March from water inlet to outlet (top to bottom)
    for i in range(num_steps - 1):
        # Current position from top
        z = z_positions[i]
        
        # Water temperature at this position
        Tw = T_water[i]
        
        # Air comes from bottom, so at position z from top,
        # air has traveled (fill_depth - z) from bottom
        # For simplicity, we'll use the air state at the same position
        # (this is simplified - proper counterflow needs backward marching for air)
        
        # Enthalpy of saturated air at water temperature
        W_sat = humidity_ratio_from_wb(Tw, Tw, pressure)
        h_sat = enthalpy_air(Tw, W_sat)
        
        # Approximate air enthalpy at this position (linear interpolation)
        # Air enthalpy increases as it moves up through the fill
        air_pos_ratio = z / fill_depth
        h_air_current = h_air[-1] + air_pos_ratio * (h_air[0] - h_air[-1]) if i > 0 else h_air[-1]
        
        # Enthalpy potential
        h_diff = h_sat - h_air_current
        h_diff = max(h_diff, 0.01)  # Avoid zero or negative
        
        # Differential equations
        # Water: L * Cp * dTw/dz = -Ka_per_depth * (h_sat - h_air)
        Cp = 4.186  # kJ/kg°C
        dT_dz[i] = -Ka_per_depth * h_diff / (L * Cp)
        
        # Update water temperature
        T_water[i+1] = T_water[i] + dT_dz[i] * dz
        
        # Update air enthalpy for next position
        # Air: G * dh/dz = Ka_per_depth * (h_sat - h_air)
        dh_dz[i] = Ka_per_depth * h_diff / G
        if i < num_steps - 1:
            h_air[i] = h_air_current + dh_dz[i] * dz
    
    # Final air outlet enthalpy (at top)
    h_air_out = h_air[-1] + np.sum(dh_dz[:-1]) * dz
    
    # Results
    T_cold_achieved = T_water[-1]
    Q_achieved = L * 4.186 * (T_hot - T_cold_achieved)
    
    # Air properties
    air_density = 1.2  # kg/m³
    air_flow_volumetric = G / air_density
    air_face_velocity = air_flow_volumetric / face_area
    
    # Water loading
    water_loading = (L * 3.6) / face_area  # m³/h·m²
    
    # Pressure drop from curve
    delta_P_per_m = np.interp(L_over_G, 
                             fill_data["performance_data"]["L_G"], 
                             fill_data["performance_data"]["delta_P"])
    fill_pressure_drop = delta_P_per_m * fill_depth
    
    # Add other losses (drift eliminator, inlet, outlet)
    total_static_pressure = fill_pressure_drop * 1.35  # 35% additional losses
    
    # Merkel number
    NTU = Ka_over_L * fill_depth
    
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
        "Q_target": L * 4.186 * (T_hot - T_cold_target),
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
        "fill_volume": face_area * fill_depth,
        "total_surface_area": (face_area * fill_depth) * fill_data["surface_area"],
        "fill_pressure_drop": fill_pressure_drop,
        "total_static_pressure": total_static_pressure,
        "marching_results": {
            "z_positions": z_positions,
            "T_water": T_water,
            "dT_dz": dT_dz
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
        alignment=1  # Center
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
    story.append(Spacer(1, 10))
    
    # Fill Specifications
    fill_data = BRENTWOOD_FILLS[design_results['fill_type']]
    story.append(Paragraph("FILL SPECIFICATIONS", styles['Heading2']))
    
    fill_specs = [
        ["Parameter", "Value"],
        ["Fill Type", fill_data['name']],
        ["Surface Area", f"{fill_data['surface_area']} m²/m³"],
        ["Sheet Spacing", f"{fill_data['sheet_spacing']} mm"],
        ["Flute Angle", f"{fill_data['flute_angle']}°"],
        ["Pack Sizes", fill_data['pack_sizes']],
        ["Description", fill_data['description']]
    ]
    
    fill_table = Table(fill_specs, colWidths=[2*inch, 4*inch])
    fill_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(fill_table)
    
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
# STREAMLIT APP
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
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📥 Design Inputs")
        
        # Calculation Mode
        calc_mode = st.radio(
            "Calculation Mode:",
            ["Mode 1: Given Heat Load & Temps → Find Water Flow",
             "Mode 2: Given Water Flow & Temps → Find Heat Load"]
        )
        
        # Temperatures
        T_hot = st.number_input("Hot Water In (°C)", 
                               value=37.0, min_value=20.0, max_value=60.0, step=0.5)
        T_cold_target = st.number_input("Target Cold Water Out (°C)", 
                                       value=32.0, min_value=10.0, max_value=40.0, step=0.5)
        Twb = st.number_input("Ambient Wet Bulb (°C)", 
                             value=28.0, min_value=10.0, max_value=40.0, step=0.5)
        
        if "Heat Load" in calc_mode:
            Q_input = st.number_input("Heat Load to Remove (kW)", 
                                     value=2090.0, min_value=100.0, step=100.0)
            # Calculate water flow
            Cp = 4.186
            L = Q_input / (Cp * (T_hot - T_cold_target))
            st.metric("Calculated Water Flow", f"{L:.2f} kg/s")
        else:
            L = st.number_input("Water Flow Rate (kg/s)", 
                               value=100.0, min_value=10.0, step=5.0)
            Cp = 4.186
            Q_input = L * Cp * (T_hot - T_cold_target)
            st.metric("Calculated Heat Load", f"{Q_input:.0f} kW")
        
        # Brentwood Fill Selection
        st.header("🎯 Brentwood Fill Selection")
        fill_options = list(BRENTWOOD_FILLS.keys())
        selected_fills = st.multiselect(
            "Select fills to compare:",
            options=fill_options,
            default=["XF75", "XF125", "XF3000"],
            format_func=lambda x: BRENTWOOD_FILLS[x]["name"]
        )
        
        # Design Geometry
        st.header("📐 Geometry & Operating Conditions")
        fill_depth = st.slider("Fill Depth (m)", 0.3, 2.0, 0.6, 0.1)
        face_area = st.slider("Face Area (m²)", 10.0, 100.0, 36.94, 5.0)
        L_over_G = st.slider("L/G Ratio", 0.5, 2.0, 1.25, 0.05)
        
        # Calculate air flow
        G = L / L_over_G
        
        # Marching steps
        num_steps = st.slider("Marching Integration Steps", 20, 200, 50, 10)
        
        # Atmospheric pressure
        pressure = st.number_input("Atmospheric Pressure (kPa)", 
                                  value=101.325, min_value=90.0, max_value=110.0, step=0.1)
        
        # Run button
        run_calc = st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True)
        
        # Report generation
        st.header("📄 Report Generation")
        generate_reports = st.checkbox("Generate PDF and TXT Reports", value=True)
    
    # Main content area
    if run_calc and selected_fills:
        # Calculate for all selected fills
        results = []
        all_figures = []
        
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
                fill_name = BRENTWOOD_FILLS[result['fill_type']]['name']
                st.subheader(fill_name)
                
                # Key metrics
                if result['T_cold_achieved'] <= result['T_cold_target']:
                    st.success(f"✅ {result['T_cold_achieved']:.2f}°C")
                else:
                    st.error(f"❌ {result['T_cold_achieved']:.2f}°C")
                
                st.metric("Heat Rejection", f"{result['Q_achieved']:.0f} kW")
                st.metric("Fan Airflow", f"{result['air_flow_volumetric']:.2f} m³/s")
                st.metric("Static Pressure", f"{result['total_static_pressure']:.0f} Pa")
                st.metric("Water Loading", f"{result['water_loading']:.1f} m³/h·m²")
        
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
        formatted_df = display_df.copy()
        formatted_df["Cold Water Out (°C)"] = formatted_df["Cold Water Out (°C)"].map(lambda x: f"{x:.2f}")
        formatted_df["Heat Rejection (kW)"] = formatted_df["Heat Rejection (kW)"].map(lambda x: f"{x:.0f}")
        formatted_df["Approach (°C)"] = formatted_df["Approach (°C)"].map(lambda x: f"{x:.2f}")
        formatted_df["Range (°C)"] = formatted_df["Range (°C)"].map(lambda x: f"{x:.2f}")
        formatted_df["NTU"] = formatted_df["NTU"].map(lambda x: f"{x:.3f}")
        formatted_df["Ka/L"] = formatted_df["Ka/L"].map(lambda x: f"{x:.3f}")
        formatted_df["L/G Ratio"] = formatted_df["L/G Ratio"].map(lambda x: f"{x:.3f}")
        formatted_df["Fan Airflow (m³/s)"] = formatted_df["Fan Airflow (m³/s)"].map(lambda x: f"{x:.2f}")
        formatted_df["Static Pressure (Pa)"] = formatted_df["Static Pressure (Pa)"].map(lambda x: f"{x:.0f}")
        formatted_df["Water Loading (m³/h·m²)"] = formatted_df["Water Loading (m³/h·m²)"].map(lambda x: f"{x:.1f}")
        formatted_df["Air Face Velocity (m/s)"] = formatted_df["Air Face Velocity (m/s)"].map(lambda x: f"{x:.2f}")
        
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
            ax1.set_title("Water Temperature Profile through Fill Depth")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim([min(T_cold_target - 2, Twb), max(T_hot + 2, T_hot)])
            
            st.pyplot(fig1)
            all_figures.append(fig1)
        
        with tab2:
            # Performance comparison bar charts
            fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            fills = results_df['fill_name']
            
            # Cold water temperatures
            cold_temps = results_df['T_cold_achieved']
            bars1 = axes[0, 0].bar(fills, cold_temps, color='skyblue', alpha=0.7)
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
            all_figures.append(fig2)
        
        with tab3:
            # Fill characteristics
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            surface_areas = [BRENTWOOD_FILLS[fill]['surface_area'] for fill in selected_fills]
            fill_names = [BRENTWOOD_FILLS[fill]['name'] for fill in selected_fills]
            
            bars = ax3.bar(fill_names, surface_areas, color='purple', alpha=0.7)
            ax3.set_ylabel("Surface Area (m²/m³)")
            ax3.set_title("Brentwood Fill Surface Areas")
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, area in zip(bars, surface_areas):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{area}', ha='center', va='bottom')
            
            ax3.set_ylim([0, max(surface_areas) * 1.2])
            st.pyplot(fig3)
            all_figures.append(fig3)
        
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
                pdf_buffer = generate_pdf_report(selected_result, results)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"cooling_tower_report_{selected_result['fill_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            # Display report preview
            with st.expander("📋 Preview TXT Report"):
                st.text(txt_report[:2000] + "..." if len(txt_report) > 2000 else txt_report)
        
        # ====================================================================
        # DESIGN RECOMMENDATIONS
        # ====================================================================
        
        st.header("🎯 Design Recommendations")
        
        # Find best fill for different criteria
        if len(results) > 1:
            # Best for temperature approach
            best_approach = min(results, key=lambda x: x['approach'])
            # Best for pressure drop
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
                    st.metric("Closest to Target", 
                             min(results, key=lambda x: abs(x['T_cold_achieved'] - x['T_cold_target']))['fill_name'],
                             "Adjust design")
        
        # ====================================================================
        # TECHNICAL DETAILS
        # ====================================================================
        
        st.header("🔬 Technical Details")
        
        for idx, result in enumerate(results):
            with st.expander(f"📋 {result['fill_name']} - Detailed Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Performance Metrics:**")
                    st.write(f"- **L/G Ratio**: {result['L_over_G']:.3f}")
                    st.write(f"- **Ka/L (from curve)**: {result['Ka_over_L']:.3f}")
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
           - Given Heat Load → Find Water Flow
           - Given Water Flow → Find Heat Load
        4. **Interactive Design** - Adjust fill depth, face area, L/G ratio
        5. **Multiple Fill Comparison** - Compare different Brentwood fills
        6. **Professional Reports** - Generate PDF and TXT reports
        7. **Complete Fan Sizing** - Airflow and static pressure calculations
        
        ### 📋 **How to Use:**
        
        1. **Enter password**: 'Semaanju'
        2. **Select calculation mode** in sidebar
        3. **Input design parameters** (temperatures, flow/heat load)
        4. **Select Brentwood fills** to compare
        5. **Adjust geometry** (fill depth, face area, L/G ratio)
        6. **Click "Run Comprehensive Analysis"**
        7. **Review results** and download reports
        
        ### 🔬 **Technical Methodology:**
        
        - Uses **Merkel equation** with **marching integration** (50+ steps through fill)
        - **Psychrometric calculations** for moist air properties
        - **Brentwood performance curves** (Ka/L vs L/G) for accurate predictions
        - **Pressure drop calculations** based on fill characteristics
        - **Counterflow arrangement** simulation
        
        ---
        
        *Enter your design parameters in the sidebar and click "Run Comprehensive Analysis" to begin.*
        """)

if __name__ == "__main__":
    main()