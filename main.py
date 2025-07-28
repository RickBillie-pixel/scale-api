import os
import re
import math
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Constants
PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Scale API v7.1.0 - Fixed Distance Rules",
    description="FIXED: Correct distance measurement rules - horizontal=midpoint, vertical=text-to-line-edge",
    version="7.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Models (Compatible with Filter API v7.0.1)
class CleanPoint(BaseModel):
    x: float
    y: float

class FilteredLine(BaseModel):
    length: float
    orientation: str
    midpoint: CleanPoint

class FilteredText(BaseModel):
    text: str
    midpoint: Dict[str, float]

class RegionData(BaseModel):
    label: str
    lines: List[FilteredLine]
    texts: List[FilteredText]
    parsed_drawing_type: Optional[str] = None

class FilteredInput(BaseModel):
    drawing_type: str
    regions: List[RegionData]

# Output Models
class ScaleCalculation(BaseModel):
    # Input data
    dimension_text: str
    dimension_value: float
    dimension_unit: str
    dimension_mm: float
    line_length_pt: float
    
    # Positions
    text_midpoint: Dict[str, float]
    line_midpoint: Dict[str, float]
    distance_text_to_line: float
    distance_method: str
    
    # Physical dimension info
    physical_dimension_type: str  # "hoogte", "breedte", "lengte"
    line_orientation: str  # "horizontal", "vertical"
    
    # Calculation details
    calculation_formula: str
    scale_pt_per_mm: float
    scale_mm_per_pt: float
    is_fallback: bool = False
    fallback_method: Optional[str] = None

class RegionScaleResult(BaseModel):
    region_label: str
    drawing_type: str
    parsed_drawing_type: Optional[str] = None
    dimension_strategy: str
    region_rules: str
    
    # Calculations per physical dimension
    vertical_calculations: List[ScaleCalculation] = []
    horizontal_calculations: List[ScaleCalculation] = []
    
    # Region averages
    total_calculations: int = 0
    average_scale_pt_per_mm: Optional[float] = None
    average_scale_mm_per_pt: Optional[float] = None
    average_calculation_formula: Optional[str] = None

class ScaleOutput(BaseModel):
    drawing_type: str
    total_regions: int
    total_calculations: int
    regions: List[RegionScaleResult]
    
    # Global average across all regions
    global_average_scale_pt_per_mm: Optional[float] = None
    global_average_scale_mm_per_pt: Optional[float] = None
    global_average_formula: Optional[str] = None
    
    # Physical dimension mapping info as string (Pydantic v2 compatible)
    physical_dimension_info: str = ""
    processing_rules: str = ""
    timestamp: str

def parse_bestektekening_region_type(region_label: str) -> str:
    """Extract drawing type from bestektekening region label"""
    
    # Check for explicit type in parentheses first
    if "(" in region_label and ")" in region_label:
        try:
            start = region_label.find("(") + 1
            end = region_label.find(")")
            extracted_type = region_label[start:end].strip()
            
            valid_types = [
                "plattegrond", "doorsnede", "gevelaanzicht", 
                "detailtekening_kozijn", "detailtekening_plattegrond",
                "detailtekening"
            ]
            
            if extracted_type in valid_types:
                return extracted_type
                
        except Exception:
            pass
    
    # Fallback to keyword matching
    label_lower = region_label.lower()
    
    if "plattegrond" in label_lower or "grond" in label_lower or "verdieping" in label_lower:
        return "plattegrond"
    elif "gevel" in label_lower or "aanzicht" in label_lower:
        return "gevelaanzicht"
    elif "doorsnede" in label_lower:
        return "doorsnede"
    elif "detail" in label_lower:
        if "kozijn" in label_lower or "raam" in label_lower or "deur" in label_lower:
            return "detailtekening_kozijn"
        else:
            return "detailtekening"
    else:
        return "unknown"

def get_physical_dimension_strategy(drawing_type: str) -> Tuple[str, Dict[str, str]]:
    """Get dimension processing strategy and physical dimension mapping"""
    
    if drawing_type == "plattegrond":
        return "process_both", {
            "horizontal_line": "breedte",
            "vertical_line": "lengte"
        }
    
    elif drawing_type == "doorsnede":
        return "process_both", {
            "vertical_line": "hoogte",
            "horizontal_line": "breedte"
        }
    
    elif drawing_type == "gevelaanzicht" or drawing_type == "gevel":
        return "vertical_only", {
            "vertical_line": "hoogte",
            "horizontal_line": "ignore"
        }
    
    elif drawing_type == "detailtekening_kozijn":
        return "process_both", {
            "horizontal_line": "breedte",
            "vertical_line": "hoogte"
        }
    
    elif drawing_type == "detailtekening_plattegrond":
        return "process_both", {
            "horizontal_line": "breedte",
            "vertical_line": "lengte"
        }
    
    elif drawing_type == "detailtekening":
        return "process_both", {
            "horizontal_line": "breedte",
            "vertical_line": "hoogte"
        }
    
    else:
        return "process_both", {
            "horizontal_line": "breedte",
            "vertical_line": "hoogte"
        }

def get_region_processing_rules(drawing_type: str, region: RegionData) -> Tuple[str, Dict[str, str], float, str]:
    """Get complete processing rules for a region - FIXED DISTANCE RULES"""
    
    # Use parsed_drawing_type for bestektekening regions
    if drawing_type == "bestektekening":
        if region.parsed_drawing_type:
            effective_drawing_type = region.parsed_drawing_type
            rules_desc = f"bestektekening_parsed_{effective_drawing_type}_rules"
        else:
            effective_drawing_type = parse_bestektekening_region_type(region.label)
            rules_desc = f"bestektekening_fallback_{effective_drawing_type}_rules"
        
        strategy, dim_mapping = get_physical_dimension_strategy(effective_drawing_type)
        
        # FIXED: Different distance thresholds per drawing type
        if effective_drawing_type == "plattegrond":
            max_distance = 15.0
        elif effective_drawing_type in ["gevelaanzicht", "doorsnede"]:
            max_distance = 15.0
        elif "detailtekening" in effective_drawing_type:
            max_distance = 10.0
        else:
            max_distance = 15.0
        
        return strategy, dim_mapping, max_distance, rules_desc
    
    # Standard drawing types
    strategy, dim_mapping = get_physical_dimension_strategy(drawing_type)
    
    if drawing_type == "plattegrond":
        max_distance = 15.0
    elif drawing_type in ["gevelaanzicht", "gevel", "doorsnede"]:
        max_distance = 15.0
    elif "detailtekening" in drawing_type:
        max_distance = 10.0
    else:
        max_distance = 15.0
    
    rules_desc = f"{drawing_type}_rules"
    return strategy, dim_mapping, max_distance, rules_desc

def calculate_midpoint_distance(text_midpoint: Dict[str, float], line_midpoint: CleanPoint) -> float:
    """Calculate Euclidean distance between text midpoint and line midpoint"""
    return math.sqrt(
        (text_midpoint["x"] - line_midpoint.x)**2 + 
        (text_midpoint["y"] - line_midpoint.y)**2
    )

def calculate_distance_to_line_edge(text_midpoint: Dict[str, float], line: FilteredLine) -> float:
    """Calculate distance from text to closest line edge (top/bottom for vertical, left/right for horizontal)"""
    
    text_x = text_midpoint["x"]
    text_y = text_midpoint["y"]
    
    if line.orientation == "vertical":
        # For vertical lines: measure to top or bottom edge
        line_top_y = line.midpoint.y + (line.length / 2)
        line_bottom_y = line.midpoint.y - (line.length / 2)
        
        distance_to_top = math.sqrt((text_x - line.midpoint.x)**2 + (text_y - line_top_y)**2)
        distance_to_bottom = math.sqrt((text_x - line.midpoint.x)**2 + (text_y - line_bottom_y)**2)
        
        return min(distance_to_top, distance_to_bottom)
        
    elif line.orientation == "horizontal":
        # For horizontal lines: measure to left or right edge
        line_left_x = line.midpoint.x - (line.length / 2)
        line_right_x = line.midpoint.x + (line.length / 2)
        
        distance_to_left = math.sqrt((text_x - line_left_x)**2 + (text_y - line.midpoint.y)**2)
        distance_to_right = math.sqrt((text_x - line_right_x)**2 + (text_y - line.midpoint.y)**2)
        
        return min(distance_to_left, distance_to_right)
    
    else:
        # Fallback to midpoint
        return calculate_midpoint_distance(text_midpoint, line.midpoint)

def calculate_distance_by_drawing_type(text_midpoint: Dict[str, float], line: FilteredLine, drawing_type: str) -> Tuple[float, str]:
    """FIXED: Calculate distance based on drawing type and line orientation rules"""
    
    # PLATTEGROND: Both orientations use midpoint-to-midpoint
    if drawing_type == "plattegrond":
        distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
        method = "midpoint_to_midpoint"
    
    # DETAILTEKENING_PLATTEGROND: Both orientations use midpoint-to-midpoint  
    elif drawing_type == "detailtekening_plattegrond":
        distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
        method = "midpoint_to_midpoint"
    
    # DETAILTEKENING_KOZIJN: Horizontal=midpoint, Vertical=text-to-edge
    elif drawing_type == "detailtekening_kozijn":
        if line.orientation == "horizontal":
            # Breedte: midpoint-to-midpoint
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
        elif line.orientation == "vertical":
            # Hoogte: text-to-line-edge (top/bottom)
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # DOORSNEDE: Horizontal=midpoint, Vertical=text-to-edge
    elif drawing_type == "doorsnede":
        if line.orientation == "horizontal":
            # Breedte: midpoint-to-midpoint
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
        elif line.orientation == "vertical":
            # Hoogte: text-to-line-edge (top/bottom)
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # GEVELAANZICHT: Only vertical, use text-to-edge
    elif drawing_type in ["gevelaanzicht", "gevel"]:
        if line.orientation == "vertical":
            # Hoogte: text-to-line-edge (top/bottom)
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            # Horizontal should be ignored, but fallback to midpoint
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # DETAILTEKENING (generic): Horizontal=midpoint, Vertical=text-to-edge
    elif drawing_type == "detailtekening":
        if line.orientation == "horizontal":
            # Breedte: midpoint-to-midpoint
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
        elif line.orientation == "vertical":
            # Hoogte: text-to-line-edge (top/bottom)
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # Fallback for unknown types
    else:
        distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
        method = "midpoint_to_midpoint"
    
    return distance, method

def extract_dimension_info(text: str) -> Optional[Tuple[float, str, float]]:
    """Extract dimension value, unit and convert to mm"""
    text = text.strip()
    
    pattern = r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$'
    match = re.match(pattern, text)
    
    if not match:
        return None
    
    try:
        value_str = match.group(1).replace(',', '.')
        value = float(value_str)
        unit = match.group(2) if match.group(2) else 'mm'
        
        conversions = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
        value_mm = value * conversions.get(unit, 1.0)
        
        return (value, unit, value_mm)
        
    except (ValueError, IndexError):
        return None

def find_dimension_line_matches(region: RegionData, strategy: str, dimension_mapping: Dict[str, str], 
                               max_distance: float, effective_drawing_type: str) -> List[Dict]:
    """Find matches based on physical dimension strategy - FIXED DISTANCE CALCULATION PER DRAWING TYPE"""
    
    all_matches = []
    
    # Get all valid dimensions
    valid_dimensions = []
    for text in region.texts:
        dim_result = extract_dimension_info(text.text)
        if dim_result:
            value, unit, value_mm = dim_result
            valid_dimensions.append({
                'text': text,
                'value': value,
                'unit': unit,
                'value_mm': value_mm
            })
    
    # Filter lines based on strategy
    if strategy == "vertical_only":
        valid_lines = [line for line in region.lines if line.orientation == "vertical"]
        allowed_orientations = ["vertical"]
    elif strategy == "process_both":
        valid_lines = region.lines
        allowed_orientations = ["horizontal", "vertical"]
    else:
        return []
    
    # Find matches with FIXED distance calculation per drawing type
    for dimension in valid_dimensions:
        for line in valid_lines:
            if line.orientation in allowed_orientations:
                line_mapping = dimension_mapping.get(f"{line.orientation}_line", "")
                if line_mapping == "ignore":
                    continue
                
                # FIXED: Use drawing-type-based distance calculation
                distance, distance_method = calculate_distance_by_drawing_type(
                    dimension['text'].midpoint, line, effective_drawing_type
                )
                
                if distance <= max_distance:
                    scale_pt_per_mm = line.length / dimension['value_mm']
                    physical_dimension_type = dimension_mapping.get(f"{line.orientation}_line", "unknown")
                    
                    match = {
                        'text': dimension['text'],
                        'line': line,
                        'dimension_value': dimension['value'],
                        'dimension_unit': dimension['unit'],
                        'dimension_mm': dimension['value_mm'],
                        'distance': distance,
                        'distance_method': distance_method,
                        'scale_pt_per_mm': scale_pt_per_mm,
                        'orientation': line.orientation,
                        'physical_dimension_type': physical_dimension_type,
                        'is_fallback': False
                    }
                    all_matches.append(match)
    
    # Sort by distance (best first)
    all_matches.sort(key=lambda x: x['distance'])
    return all_matches

def try_fallback_strategies(region: RegionData, effective_drawing_type: str, dimension_mapping: Dict[str, str]) -> List[Dict]:
    """Try fallback strategies when primary matching fails - FIXED DISTANCE RULES PER DRAWING TYPE"""
    
    fallback_matches = []
    
    if effective_drawing_type in ["gevelaanzicht", "gevel", "doorsnede"]:
        
        # Fallback 1: Increased distance with correct drawing-type-based method
        for fallback_distance in [25.0, 40.0]:
            matches = find_dimension_line_matches(
                region, "process_both" if effective_drawing_type == "doorsnede" else "vertical_only", 
                dimension_mapping, fallback_distance, effective_drawing_type
            )
            if matches:
                for match in matches[:3]:
                    match['is_fallback'] = True
                    match['fallback_method'] = f"increased_distance_{fallback_distance}pt"
                    fallback_matches.extend(matches[:3])
                break
        
        # Fallback 2: Even higher distance with drawing-type-based method
        if not fallback_matches:
            matches = find_dimension_line_matches(
                region, "process_both" if effective_drawing_type == "doorsnede" else "vertical_only",
                dimension_mapping, 60.0, effective_drawing_type
            )
            if matches:
                for match in matches[:3]:
                    match['is_fallback'] = True
                    match['fallback_method'] = "very_increased_distance_60pt"
                    fallback_matches.extend(matches[:3])
        
        # Fallback 3: Reference calculation for gevel
        if not fallback_matches and effective_drawing_type in ["gevelaanzicht", "gevel"]:
            reference_match = try_total_height_reference(region, dimension_mapping)
            if reference_match:
                fallback_matches.append(reference_match)
    
    return fallback_matches

def try_total_height_reference(region: RegionData, dimension_mapping: Dict[str, str]) -> Optional[Dict]:
    """Try total height reference calculation for gevel"""
    try:
        # Find highest dimension (total building height)
        all_dimensions = []
        for text in region.texts:
            dim_result = extract_dimension_info(text.text)
            if dim_result:
                value, unit, value_mm = dim_result
                all_dimensions.append((value_mm, text, value, unit))
        
        if not all_dimensions:
            return None
        
        # Get highest dimension
        max_dimension_mm, max_text, max_value, max_unit = max(all_dimensions)
        
        # Find longest vertical line (total building height)
        vertical_lines = [line for line in region.lines if line.orientation == "vertical"]
        if not vertical_lines:
            return None
        
        longest_line = max(vertical_lines, key=lambda x: x.length)
        
        # Create reference calculation
        reference_scale = longest_line.length / max_dimension_mm
        
        return {
            'text': max_text,
            'line': longest_line,
            'dimension_value': max_value,
            'dimension_unit': max_unit,
            'dimension_mm': max_dimension_mm,
            'distance': 999.0,
            'distance_method': 'reference_calculation',
            'scale_pt_per_mm': reference_scale,
            'orientation': 'vertical',
            'physical_dimension_type': 'hoogte',
            'is_fallback': True,
            'fallback_method': 'total_height_reference'
        }
        
    except Exception:
        return None

def select_best_matches_per_orientation(all_matches: List[Dict], max_per_orientation: int = 3) -> Tuple[List[Dict], List[Dict]]:
    """Select best matches per orientation, avoiding duplicates"""
    
    used_texts = set()
    used_lines = set()
    horizontal_matches = []
    vertical_matches = []
    
    # First pass: collect horizontal matches
    for match in all_matches:
        if (len(horizontal_matches) < max_per_orientation and 
            match['orientation'] == 'horizontal' and
            id(match['text']) not in used_texts and 
            id(match['line']) not in used_lines):
            
            horizontal_matches.append(match)
            used_texts.add(id(match['text']))
            used_lines.add(id(match['line']))
    
    # Second pass: collect vertical matches
    for match in all_matches:
        if (len(vertical_matches) < max_per_orientation and 
            match['orientation'] == 'vertical' and
            id(match['text']) not in used_texts and 
            id(match['line']) not in used_lines):
            
            vertical_matches.append(match)
            used_texts.add(id(match['text']))
            used_lines.add(id(match['line']))
    
    return horizontal_matches, vertical_matches

def create_scale_calculation(match: Dict) -> ScaleCalculation:
    """Create ScaleCalculation object from match data"""
    
    scale_pt_per_mm = match['scale_pt_per_mm']
    scale_mm_per_pt = 1 / scale_pt_per_mm
    
    formula_suffix = " (reference)" if match.get('is_fallback') and "reference" in match.get('fallback_method', '') else ""
    formula = f"{match['line'].length:.1f}pt ÷ {match['dimension_mm']:.0f}mm = {scale_pt_per_mm:.4f} pt/mm{formula_suffix}"
    
    return ScaleCalculation(
        dimension_text=match['text'].text,
        dimension_value=match['dimension_value'],
        dimension_unit=match['dimension_unit'],
        dimension_mm=match['dimension_mm'],
        line_length_pt=match['line'].length,
        text_midpoint=match['text'].midpoint,
        line_midpoint={"x": match['line'].midpoint.x, "y": match['line'].midpoint.y},
        distance_text_to_line=round(match['distance'], 2),
        distance_method=match['distance_method'],
        physical_dimension_type=match['physical_dimension_type'],
        line_orientation=match['orientation'],
        calculation_formula=formula,
        scale_pt_per_mm=round(scale_pt_per_mm, 4),
        scale_mm_per_pt=round(scale_mm_per_pt, 4),
        is_fallback=match.get('is_fallback', False),
        fallback_method=match.get('fallback_method')
    )

def process_region_with_physical_dimensions(region: RegionData, drawing_type: str) -> RegionScaleResult:
    """Process a region using parsed drawing type - FIXED DISTANCE RULES"""
    
    # Get processing rules for this region
    strategy, dimension_mapping, max_distance, rules_desc = get_region_processing_rules(
        drawing_type, region
    )
    
    # Determine effective drawing type
    if drawing_type == "bestektekening" and region.parsed_drawing_type:
        effective_drawing_type = region.parsed_drawing_type
    elif drawing_type == "bestektekening":
        effective_drawing_type = parse_bestektekening_region_type(region.label)
    else:
        effective_drawing_type = drawing_type
    
    result = RegionScaleResult(
        region_label=region.label,
        drawing_type=drawing_type,
        parsed_drawing_type=region.parsed_drawing_type,
        dimension_strategy=strategy,
        region_rules=rules_desc
    )
    
    # Primary matching attempt with FIXED distance calculation per drawing type
    all_matches = find_dimension_line_matches(region, strategy, dimension_mapping, max_distance, effective_drawing_type)
    
    # Try fallbacks if no matches found
    if not all_matches:
        all_matches = try_fallback_strategies(region, effective_drawing_type, dimension_mapping)
    
    if not all_matches:
        return result
    
    # Select best matches per orientation
    horizontal_matches, vertical_matches = select_best_matches_per_orientation(all_matches, 3)
    
    # Create scale calculations
    result.horizontal_calculations = [create_scale_calculation(match) for match in horizontal_matches]
    result.vertical_calculations = [create_scale_calculation(match) for match in vertical_matches]
    
    # Calculate region averages
    all_scale_values = [match['scale_pt_per_mm'] for match in horizontal_matches + vertical_matches]
    result.total_calculations = len(all_scale_values)
    
    if all_scale_values:
        avg_scale = sum(all_scale_values) / len(all_scale_values)
        result.average_scale_pt_per_mm = round(avg_scale, 4)
        result.average_scale_mm_per_pt = round(1 / avg_scale, 4)
        
        result.average_calculation_formula = f"({' + '.join([f'{s:.4f}' for s in all_scale_values])}) ÷ {len(all_scale_values)} = {avg_scale:.4f} pt/mm"
    
    return result

@app.post("/calculate-scale/", response_model=ScaleOutput)
async def calculate_scale(input_data: FilteredInput):
    """Calculate scale with FIXED distance measurement rules"""
    try:
        if input_data.drawing_type == "installatietekening":
            # Skip installatietekening
            return ScaleOutput(
                drawing_type=input_data.drawing_type,
                total_regions=0,
                total_calculations=0,
                regions=[],
                physical_dimension_info='{"installatietekening": "skipped"}',
                processing_rules='{"installatietekening": "skipped"}',
                timestamp=datetime.now().isoformat()
            )
        
        region_results = []
        all_regional_scales = []
        total_calculations = 0
        processing_rules = {}
        
        # Get physical dimension info for this drawing type
        strategy, dimension_mapping = get_physical_dimension_strategy(input_data.drawing_type)
        physical_dimension_info = {
            "drawing_type": input_data.drawing_type,
            "strategy": strategy,
            "dimension_mapping": dimension_mapping,
            "distance_rules": {
                "horizontal_lines": "midpoint_to_midpoint (for breedte/lengte)",
                "vertical_lines": "text_to_line_edge (for hoogte measurements)"
            }
        }
        
        # Process each region
        for region in input_data.regions:
            if region.lines and region.texts:
                region_result = process_region_with_physical_dimensions(region, input_data.drawing_type)
                region_results.append(region_result)
                
                # Collect scales for global average
                if region_result.average_scale_pt_per_mm:
                    all_regional_scales.append(region_result.average_scale_pt_per_mm)
                    total_calculations += region_result.total_calculations
                
                # Track processing rules
                processing_rules[region.label] = region_result.region_rules
        
        # Calculate global averages
        global_avg_scale = None
        global_avg_scale_mm = None
        global_formula = None
        
        if all_regional_scales:
            global_avg_scale = sum(all_regional_scales) / len(all_regional_scales)
            global_avg_scale_mm = 1 / global_avg_scale
            global_formula = f"({' + '.join([f'{s:.4f}' for s in all_regional_scales])}) ÷ {len(all_regional_scales)} = {global_avg_scale:.4f} pt/mm"
        
        # Convert dict to JSON string for Pydantic v2 compatibility
        import json
        
        # Build response
        response = ScaleOutput(
            drawing_type=input_data.drawing_type,
            total_regions=len(region_results),
            total_calculations=total_calculations,
            regions=region_results,
            global_average_scale_pt_per_mm=round(global_avg_scale, 4) if global_avg_scale else None,
            global_average_scale_mm_per_pt=round(global_avg_scale_mm, 4) if global_avg_scale_mm else None,
            global_average_formula=global_formula,
            physical_dimension_info=json.dumps(physical_dimension_info),
            processing_rules=json.dumps(processing_rules),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scale calculation error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Scale API v7.1.0 - Fixed Distance Rules",
        "version": "7.1.0",
        "description": "FIXED: Correct distance measurement rules implemented",
        "bug_fix": {
            "issue": "Inconsistent distance measurement methods causing scale variations",
            "solution": "Fixed orientation-based distance rules: horizontal=midpoint, vertical=text-to-line-edge",
            "improvement": "Should significantly improve scale consistency"
        },
        "distance_rules": {
            "horizontal_lines": {
                "method": "midpoint_to_midpoint",
                "reason": "For breedte and lengte measurements",
                "usage": "Always used for horizontal line orientation"
            },
            "vertical_lines": {
                "method": "text_to_line_edge", 
                "reason": "For hoogte measurements (top/bottom edges)",
                "usage": "Always used for vertical line orientation"
            }
        },
        "physical_dimension_mapping": {
            "plattegrond": {
                "horizontal_line": "breedte (midpoint method)",
                "vertical_line": "lengte (midpoint method)",
                "strategy": "process_both",
                "max_distance": "15pt"
            },
            "detailtekening_plattegrond": {
                "horizontal_line": "breedte (midpoint method)",
                "vertical_line": "lengte (midpoint method)", 
                "strategy": "process_both",
                "max_distance": "10pt"
            },
            "detailtekening_kozijn": {
                "horizontal_line": "breedte (midpoint method)",
                "vertical_line": "hoogte (text-to-edge method)",
                "strategy": "process_both", 
                "max_distance": "10pt"
            },
            "doorsnede": {
                "vertical_line": "hoogte (text-to-edge method)",
                "horizontal_line": "breedte (midpoint method)",
                "strategy": "process_both",
                "max_distance": "15pt"
            },
            "gevelaanzicht": {
                "vertical_line": "hoogte (text-to-edge method)",
                "horizontal_line": "IGNORE",
                "strategy": "vertical_only",
                "max_distance": "15pt"
            },
            "detailtekening": {
                "horizontal_line": "breedte (midpoint method)",
                "vertical_line": "hoogte (text-to-edge method)",
                "strategy": "process_both",
                "max_distance": "10pt"
            }
        },
        "improvements": {
            "consistency": "Fixed orientation-based distance calculation",
            "accuracy": "Proper text-to-line-edge for vertical measurements",
            "reliability": "Consistent midpoint calculation for horizontal measurements"
        },
        "endpoints": {
            "/calculate-scale/": "Main scale calculation endpoint (FIXED)",
            "/health/": "Health check with updated feature status",
            "/": "This comprehensive documentation"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "7.1.0",
        "port": PORT,
        "distance_rules_fixed": True,
        "features": {
            "orientation_based_distance": True,
            "horizontal_midpoint_method": True,
            "vertical_text_to_edge_method": True,
            "consistent_scale_calculation": True,
            "filter_api_v7_compatibility": True,
            "physical_dimension_mapping": True,
            "bestektekening_support": True,
            "fallback_strategies": True,
            "pydantic_v2_serialization": True
        },
        "supported_drawing_types": [
            "plattegrond", "doorsnede", "gevelaanzicht",
            "detailtekening_kozijn", "detailtekening_plattegrond", 
            "detailtekening", "bestektekening"
        ],
        "skipped_types": ["installatietekening"],
        "distance_methods": {
            "plattegrond": "Both orientations: midpoint_to_midpoint",
            "detailtekening_plattegrond": "Both orientations: midpoint_to_midpoint", 
            "detailtekening_kozijn": "Horizontal: midpoint | Vertical: text-to-edge",
            "doorsnede": "Horizontal: midpoint | Vertical: text-to-edge", 
            "gevelaanzicht": "Vertical only: text-to-edge",
            "detailtekening": "Horizontal: midpoint | Vertical: text-to-edge",
            "threshold_distances": {
                "plattegrond": "15pt",
                "gevelaanzicht": "15pt", 
                "doorsnede": "15pt",
                "detailtekening": "10pt",
                "detailtekening_plattegrond": "10pt",
                "detailtekening_kozijn": "10pt"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
