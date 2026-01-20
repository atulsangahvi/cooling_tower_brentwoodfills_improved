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
    
    # ... [previous sections remain the same] ...
    
    # NEW SECTION: MANUFACTURER SPECIFICATIONS
    report.append("\nMANUFACTURER SPECIFICATIONS")
    report.append("-" * 40)
    report.append(f"Maximum Water Loading: {design_results['max_water_loading']} m³/h·m²")
    report.append(f"Minimum Water Loading: {design_results['min_water_loading']} m³/h·m²")
    report.append(f"Maximum Air Velocity: {design_results['max_air_velocity']} m/s")
    report.append(f"Recommended Air Velocity: {design_results['recommended_air_velocity']} m/s")
    
    # Check against limits
    report.append("\nLIMIT COMPLIANCE CHECK")
    report.append("-" * 40)
    
    water_status = "EXCEEDS" if design_results['water_loading'] > design_results['max_water_loading'] else \
                  "BELOW" if design_results['water_loading'] < design_results['min_water_loading'] else "WITHIN"
    
    air_status = "EXCEEDS" if design_results['air_face_velocity'] > design_results['max_air_velocity'] else "WITHIN"
    
    report.append(f"Water Loading: {design_results['water_loading']:.1f} m³/h·m² → {water_status} limits")
    report.append(f"Air Velocity: {design_results['air_face_velocity']:.2f} m/s → {air_status} limits")
    
    # ... [rest of the report remains the same] ...