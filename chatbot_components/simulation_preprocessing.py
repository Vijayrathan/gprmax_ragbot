
def parse_gprmax_input(file_path):
    """Dynamically extracts all parameters rather than looking for specific ones
        Properly handles Python code blocks
        Extracts parameters from Python variable assignments and function calls
        Identifies geometry elements in both direct commands and Python code"""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Initialize empty parameters dictionary
    params = {}
    # Track Python blocks
    in_python_block = False
    python_code = []
    # Additional comments for reference
    reference_comments = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Handle comment lines (double hash ##)
        if line.startswith("##"):
            reference_comments.append(line[2:].strip())
            continue
            
        # Handle Python code blocks
        if line == "#python:":
            in_python_block = True
            python_code = []
            continue
        elif line == "#end_python:":
            in_python_block = False
            params["python_block"] = "\n".join(python_code)
            
            # Try to extract important Python-defined parameters
            if "python_block" in params:
                python_code_str = params["python_block"]
                import re
                
                # Look for common parameter definitions in Python code
                if "title =" in python_code_str:
                    title_match = re.search(r"title\s*=\s*['\"]([^'\"]+)['\"]", python_code_str)
                    if title_match:
                        params["title"] = title_match.group(1)
                
                # Extract domain, dx_dy_dz, time_window from Python code
                if "domain" in python_code_str:
                    domain_match = re.search(r"domain\s*=\s*domain\(([^)]+)\)", python_code_str)
                    if domain_match:
                        params["domain"] = domain_match.group(1).replace(",", " ")
                
                if "dx_dy_dz" in python_code_str or "dxdydz" in python_code_str:
                    dxdydz_match = re.search(r"(?:dxdydz|dx_dy_dz)\s*=\s*dx_dy_dz\(([^)]+)\)", python_code_str)
                    if dxdydz_match:
                        params["dx_dy_dz"] = dxdydz_match.group(1).replace(",", " ")
                
                if "time_window" in python_code_str:
                    time_window_match = re.search(r"time_window\s*=\s*time_window\(([^)]+)\)", python_code_str)
                    if time_window_match:
                        params["time_window"] = time_window_match.group(1)
                
                # Look for print statements with GPRMax commands
                print_commands = re.findall(r"print\(['\"](#[^:]+:[^'\"]+)['\"](?:\.format\(([^)]+)\))?", python_code_str)
                for cmd_match in print_commands:
                    cmd_text = cmd_match[0]
                    format_vars = cmd_match[1] if len(cmd_match) > 1 else ""
                    
                    # Parse the command
                    parts = cmd_text.split(":", 1)
                    if len(parts) == 2:
                        cmd_name = parts[0][1:]  # Remove the # character
                        value = parts[1].strip()
                        
                        # If format is used with a variable, try to find the variable's value
                        if format_vars and value == "{}":
                            # If the format variable is 'title', use the title we already extracted
                            if format_vars.strip() == "title" and "title" in params:
                                value = params["title"]
                        
                        params[cmd_name] = value
                
                # Look for function calls that generate geometry
                geometry_funcs = ["box", "cylinder", "triangle", "sphere", "plate", "geometry_view", "edge", "wedge"]
                for func in geometry_funcs:
                    func_pattern = r"{}".format(func) + r"\s*\([^)]+\)"
                    geo_matches = re.findall(func_pattern, python_code_str)
                    if geo_matches:
                        if "python_geometry" not in params:
                            params["python_geometry"] = []
                        for match in geo_matches:
                            # Extract the actual call
                            params["python_geometry"].append(f"Uses {func}: {match[:50]}...")
            
            continue
        
        # Collect Python code
        if in_python_block:
            python_code.append(line)
            continue
            
        # For regular commands that start with #
        if line.startswith("#"):
            # Extract command and rest of the line
            parts = line.split(":", 1)
            if len(parts) == 2:
                cmd = parts[0][1:]  # Remove the # character
                value = parts[1].strip()
                
                # Handle the special case for geometry items
                if cmd in ["box", "cylinder", "triangle", "sphere", "plate", "geometry_view", "edge", "wedge"]:
                    if "geometry" not in params:
                        params["geometry"] = []
                    params["geometry"].append(line)
                else:
                    # Store the parameter
                    params[cmd] = value
        # For parameters without # (like the pml_cfs lines after reference)
        elif ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                cmd = parts[0].strip()
                value = parts[1].strip()
                
                # Handle repeated parameters (like multiple pml_cfs lines)
                if cmd in params:
                    # If this is the first duplicate, convert to list
                    if not isinstance(params[cmd], list):
                        params[cmd] = [params[cmd]]
                    # Add the new value
                    params[cmd].append(value)
                else:
                    # First occurrence
                    params[cmd] = value
    
    # Add reference comments if any were collected
    if reference_comments:
        params["reference_comments"] = reference_comments
    
    return params

def simulation_description(params):
    """Generate a descriptive text from simulation parameters dynamically"""
    description = []
    
    # Clean up parameter values to remove format placeholders
    cleaned_params = {}
    for key, value in params.items():
        if isinstance(value, str) and '{' in value and '}' in value:
            # This is likely a format string that wasn't interpolated
            # Replace it with a more generic description
            if key == 'transmission_line':
                cleaned_params[key] = "defined in code"
            else:
                cleaned_params[key] = "dynamically defined"
        else:
            cleaned_params[key] = value
    
    # Add title if available
    if "title" in cleaned_params:
        description.append(f"Simulation titled '{cleaned_params['title']}'")
    
    # Add domain if available
    if "domain" in cleaned_params:
        description.append(f"with domain {cleaned_params['domain']} m")
    
    # Add resolution if available
    if "dx_dy_dz" in cleaned_params:
        description.append(f"resolution {cleaned_params['dx_dy_dz']} m")
    
    # Add time window if available
    if "time_window" in cleaned_params:
        description.append(f"and time window {cleaned_params['time_window']} s")
    
    # Add material definitions if available
    if "material" in cleaned_params:
        description.append(f"Material defined as {cleaned_params['material']}")
    
    # Add waveform if available
    if "waveform" in cleaned_params:
        description.append(f"Waveform used: {cleaned_params['waveform']}")
    
    # Add source information
    source_types = ["hertzian_dipole", "magnetic_dipole", "transmission_line"]
    sources = []
    for src_type in source_types:
        if src_type in cleaned_params:
            sources.append(f"{src_type}: {cleaned_params[src_type]}")
    
    if sources:
        description.append(f"Source(s): {'; '.join(sources)}")
    
    # Add receiver information
    if "rx" in cleaned_params:
        description.append(f"Receiver at {cleaned_params['rx']}")
    
    # Add source/receiver steps if available
    if "src_steps" in cleaned_params:
        description.append(f"Source steps: {cleaned_params['src_steps']}")
    if "rx_steps" in cleaned_params:
        description.append(f"Receiver steps: {cleaned_params['rx_steps']}")
    
    # Add geometry information
    if "geometry" in cleaned_params and cleaned_params["geometry"]:
        # Truncate long geometry descriptions
        short_geo = []
        for geo in cleaned_params["geometry"]:
            if len(geo) > 50:
                short_geo.append(geo[:50] + "...")
            else:
                short_geo.append(geo)
        description.append(f"Geometry includes: {'; '.join(short_geo)}")
    
    # Add Python-generated geometry information
    if "python_geometry" in cleaned_params and cleaned_params["python_geometry"]:
        description.append(f"Python-generated geometry: {'; '.join(cleaned_params['python_geometry'])}")
    
    # Add PML information
    if "pml_cells" in cleaned_params:
        description.append(f"PML cells: {cleaned_params['pml_cells']}")
    if "pml_formulation" in cleaned_params:
        description.append(f"PML formulation: {cleaned_params['pml_formulation']}")
    
    # Handle multiple PML CFS settings
    if "pml_cfs" in cleaned_params:
        if isinstance(cleaned_params["pml_cfs"], list):
            cfs_descriptions = []
            for i, cfs in enumerate(cleaned_params["pml_cfs"]):
                if i == 0:
                    cfs_descriptions.append(f"Built-in PML: {cfs}")
                else:
                    cfs_descriptions.append(f"Additional PML scheme {i}: {cfs}")
            description.append(f"PML conductivity factor schemes: {'; '.join(cfs_descriptions)}")
        else:
            description.append(f"PML conductivity factor scheme: {cleaned_params['pml_cfs']}")
    
    # Add reference comments if available
    if "reference_comments" in cleaned_params:
        ref_text = " ".join(cleaned_params["reference_comments"])
        if len(ref_text) > 100:
            ref_text = ref_text[:100] + "..."
        description.append(f"References: {ref_text}")
    
    # Add Python block information if available
    if "python_block" in cleaned_params:
        description.append("Contains custom Python code for advanced configuration")
    
    # Add additional parameters that weren't explicitly handled
    additional_params = []
    for key, value in cleaned_params.items():
        if key not in ["title", "domain", "dx_dy_dz", "time_window", "material", 
                      "waveform", "hertzian_dipole", "magnetic_dipole", "transmission_line", 
                      "rx", "src_steps", "rx_steps", "geometry", "python_block", "python_geometry",
                      "pml_cells", "pml_formulation", "pml_cfs", "reference_comments"]:
            additional_params.append(f"{key}: {value}")
    
    if additional_params:
        description.append(f"Additional parameters: {'; '.join(additional_params)}")
    
    # If the description is empty but we have Python code, add a basic description
    if not description and "python_block" in cleaned_params:
        description.append("GPRMax simulation file with Python-generated configuration")
    
    # Combine all description parts with proper spacing
    return ". ".join(description) + "."
