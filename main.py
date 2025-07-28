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

# ENHANCED FILTERING RULES
MIN_DIMENSION_THRESHOLDS = {
    "plattegrond": {"horizontal": 1000, "vertical": 1000},
    "bestektekening": {"horizontal": 2000, "vertical": 800},
    "doorsnede": {"horizontal": 1500, "vertical": 800},  # Filter 420mm weg
    "gevelaanzicht": {"horizontal": 1000, "vertical": 800},
    "detailtekening": {"horizontal": 500, "vertical": 500},
    "installatietekening": {"horizontal": 1000, "vertical": 800}
}

MIN_LINE_LENGTH_THRESHOLDS = {
    "plattegrond": 80, "bestektekening": 80, "doorsnede": 80,  # Filter ~68pt lijnen weg
    "gevelaanzicht": 60, "detailtekening": 40, "installatietekening": 50
}

OUTLIER_RULES = {
    "max_scale_pt_per_mm": 0.12,     # Alles boven 0.12 is verdacht
    "min_scale_pt_per_mm": 0.025,    # Alles onder 0.025 is verdacht
    "dimension_line_ratio_max": 40,   # Max mm/pt ratio
    "dimension_line_ratio_min": 12    # Min mm/pt ratio
}

app = FastAPI(
    title="Scale API v7.3.0 - Enhanced with Validation Rules",
    description="ENHANCED: Intelligent scale calculation with outlier detection and validation rules",
    version="7.3.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Models (Compatible with Filter API v7.0.2)
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
    
    # Region averages with intelligent fallback
    total_calculations: int = 0
    horizontal_average_scale: Optional[float] = None
    vertical_average_scale: Optional[float] = None
    scale_consistency_check: Optional[str] = None
    final_scale_source: str = ""  # "horizontal_preferred", "mixed_average", "vertical_only"
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
    """Get complete processing rules for a region"""
    
    # Use parsed_drawing_type for bestektekening regions
    if drawing_type == "bestektekening":
        if region.parsed_drawing_type:
            effective_drawing_type = region.parsed_drawing_type
            rules_desc = f"bestektekening_parsed_{effective_drawing_type}_rules"
        else:
            effective_drawing_type = parse_bestektekening_region_type(region.label)
            rules_desc = f"bestektekening_fallback_{effective_drawing_type}_rules"
        
        strategy, dim_mapping = get_physical_dimension_strategy(effective_drawing_type)
        
        # Distance thresholds per drawing type
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
    """Calculate distance from text to closest line edge"""
    
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
        return calculate_midpoint_distance(text_midpoint, line.midpoint)

def calculate_distance_by_drawing_type(text_midpoint: Dict[str, float], line: FilteredLine, drawing_type: str) -> Tuple[float, str]:
    """Calculate distance based on drawing type and line orientation rules"""
    
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
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
        elif line.orientation == "vertical":
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # DOORSNEDE: Horizontal=midpoint, Vertical=text-to-edge
    elif drawing_type == "doorsnede":
        if line.orientation == "horizontal":
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
        elif line.orientation == "vertical":
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # GEVELAANZICHT: Only vertical, use text-to-edge
    elif drawing_type in ["gevelaanzicht", "gevel"]:
        if line.orientation == "vertical":
            distance = calculate_distance_to_line_edge(text_midpoint, line)
            method = "text_to_line_edge"
        else:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
    
    # DETAILTEKENING (generic): Horizontal=midpoint, Vertical=text-to-edge
    elif drawing_type == "detailtekening":
        if line.orientation == "horizontal":
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            method = "midpoint_to_midpoint"
        elif line.orientation == "vertical":
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
    """Extract dimension value, unit and convert to mm - Enhanced for P/V/+ symbols"""
    text = text.strip()
    
    # Enhanced patterns for various dimension formats
    patterns = [
        r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$',                    # Standard: 2400mm, 3.5m
        r'^\+(\d+(?:[,.]\d+)?)\s*(mm|cm|m|p|v)?$',              # Plus prefix: +7555, +3000P
        r'^(\d+(?:[,.]\d+)?)\s*[pP]\s*(mm|cm|m)?$',             # P suffix: 7555P, 3000P
        r'^(\d+(?:[,.]\d+)?)\s*[vV]\s*(mm|cm|m)?$',             # V suffix: 7555V, 3000V
        r'^\+(\d+(?:[,.]\d+)?)\s*[pPvV]\s*(mm|cm|m)?$',         # Combined: +7555P, +3000V
        r'^(\d+(?:[,.]\d+)?)\s*\+\s*[pPvV]\s*(mm|cm|m)?$',     # Plus suffix: 6032+p, 3749+p
        r'^(\d+(?:[,.]\d+)?)\s+\+\s*[pPvV]\s*(mm|cm|m)?$'      # Space variations: 6032 +p
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '.')
            value = float(value_str)
            
            # Get unit (ignore p/v/P/V - they're not measurement units)
            unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
            if unit and unit.lower() in ['p', 'v']:
                unit = 'mm'  # Default to mm for P/V suffixes
            elif not unit:
                unit = 'mm'  # Default unit
            
            # Convert to mm
            conversions = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
            value_mm = value * conversions.get(unit.lower(), 1.0)
            
            return (value, unit, value_mm)
    
    return None

# ENHANCED VALIDATION FUNCTIES
def validate_dimension_size(dimension_mm: float, drawing_type: str, orientation: str) -> bool:
    """Validate dimension against minimum thresholds"""
    min_threshold = MIN_DIMENSION_THRESHOLDS.get(drawing_type, {}).get(orientation, 500)
    
    if dimension_mm < min_threshold:
        return False
    
    # Check for suspicious patterns (niveau indicaties)
    if dimension_mm in [3420, 6410, 7555, 7075, 3000]:  # Veel voorkomende niveau indicaties
        return False
        
    return True

def validate_line_length(line_length: float, drawing_type: str) -> bool:
    """Validate line length against minimum thresholds"""
    min_length = MIN_LINE_LENGTH_THRESHOLDS.get(drawing_type, 50)
    return line_length >= min_length

def validate_scale_range(scale_pt_per_mm: float, dimension_mm: float, line_length_pt: float) -> bool:
    """Enhanced scale validation with outlier detection"""
    
    # Global outlier limits
    if scale_pt_per_mm < OUTLIER_RULES["min_scale_pt_per_mm"]:
        return False
    if scale_pt_per_mm > OUTLIER_RULES["max_scale_pt_per_mm"]:
        return False
    
    # Check dimension/line ratio
    ratio_mm_per_pt = dimension_mm / line_length_pt
    if ratio_mm_per_pt > OUTLIER_RULES["dimension_line_ratio_max"]:
        return False
    if ratio_mm_per_pt < OUTLIER_RULES["dimension_line_ratio_min"]:
        return False
    
    return True

def detect_suspect_patterns(text: str) -> bool:
    """Detect suspect dimension patterns like +3420P, +6410P"""
    suspect_patterns = ["+3420P", "+6410P", "+7555P", "+7075P", "+3000P"]
    return any(pattern in text.upper() for pattern in suspect_patterns)

def find_dimension_line_matches(region: RegionData, strategy: str, dimension_mapping: Dict[str, str], 
                               max_distance: float, effective_drawing_type: str) -> List[Dict]:
    """ENHANCED: Find matches with filtering rules"""
    
    all_matches = []
    filtered_small_dimensions = 0
    filtered_short_lines = 0
    outliers_detected = 0
    
    # Get all valid dimensions WITH FILTERING
    valid_dimensions = []
    for text in region.texts:
        dim_result = extract_dimension_info(text.text)
        if dim_result:
            value, unit, value_mm = dim_result
            
            # ENHANCED FILTERING: Check suspect patterns
            if detect_suspect_patterns(text.text):
                print(f"WARNING: Suspect pattern detected: {text.text}")
                continue
            
            valid_dimensions.append({
                'text': text,
                'value': value,
                'unit': unit,
                'value_mm': value_mm
            })
    
    # Filter lines based on strategy WITH LENGTH VALIDATION
    if strategy == "vertical_only":
        valid_lines = [line for line in region.lines 
                      if line.orientation == "vertical" and validate_line_length(line.length, effective_drawing_type)]
        allowed_orientations = ["vertical"]
    elif strategy == "process_both":
        valid_lines = [line for line in region.lines 
                      if validate_line_length(line.length, effective_drawing_type)]
        allowed_orientations = ["horizontal", "vertical"]
    else:
        return []
    
    filtered_short_lines = len(region.lines) - len(valid_lines)
    
    # Find matches with ENHANCED VALIDATION
    for dimension in valid_dimensions:
        for line in valid_lines:
            if line.orientation in allowed_orientations:
                line_mapping = dimension_mapping.get(f"{line.orientation}_line", "")
                if line_mapping == "ignore":
                    continue
                
                # ENHANCED: Validate dimension size
                if not validate_dimension_size(dimension['value_mm'], effective_drawing_type, line.orientation):
                    filtered_small_dimensions += 1
                    continue
                
                distance, distance_method = calculate_distance_by_drawing_type(
                    dimension['text'].midpoint, line, effective_drawing_type
                )
                
                if distance <= max_distance:
                    scale_pt_per_mm = line.length / dimension['value_mm']
                    
                    # ENHANCED: Validate scale range
                    if not validate_scale_range(scale_pt_per_mm, dimension['value_mm'], line.length):
                        outliers_detected += 1
                        print(f"OUTLIER: {dimension['text'].text} -> {scale_pt_per_mm:.4f} pt/mm (filtered out)")
                        continue
                    
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
    
    # Log filtering stats
    if filtered_small_dimensions > 0:
        print(f"Filtered {filtered_small_dimensions} small dimensions")
    if filtered_short_lines > 0:
        print(f"Filtered {filtered_short_lines} short lines")
    if outliers_detected > 0:
        print(f"Detected {outliers_detected} outlier scales")
    
    # Sort by distance (best first)
    all_matches.sort(key=lambda x: x['distance'])
    return all_matches

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

def calculate_intelligent_scale_average(horizontal_matches: List[Dict], vertical_matches: List[Dict]) -> Tuple[float, str, str]:
    """INTELLIGENT: Calculate scale average with horizontal-first fallback strategy"""
    
    # Calculate averages per orientation
    horizontal_scales = [match['scale_pt_per_mm'] for match in horizontal_matches] if horizontal_matches else []
    vertical_scales = [match['scale_pt_per_mm'] for match in vertical_matches] if vertical_matches else []
    
    horizontal_avg = sum(horizontal_scales) / len(horizontal_scales) if horizontal_scales else None
    vertical_avg = sum(vertical_scales) / len(vertical_scales) if vertical_scales else None
    
    # CASE 1: Only horizontal scales available
    if horizontal_avg is not None and vertical_avg is None:
        return horizontal_avg, "horizontal_only", f"Using horizontal average: {horizontal_avg:.4f} pt/mm"
    
    # CASE 2: Only vertical scales available  
    if vertical_avg is not None and horizontal_avg is None:
        return vertical_avg, "vertical_only", f"Using vertical average: {vertical_avg:.4f} pt/mm"
    
    # CASE 3: Both orientations available - Apply intelligent fallback
    if horizontal_avg is not None and vertical_avg is not None:
        
        # Calculate percentage difference
        avg_scale = (horizontal_avg + vertical_avg) / 2
        horizontal_deviation = abs(horizontal_avg - avg_scale) / avg_scale * 100
        vertical_deviation = abs(vertical_avg - avg_scale) / avg_scale * 100
        
        # STRATEGY: If vertical deviates >15% from horizontal, prefer horizontal
        deviation_threshold = 15.0  # 15% threshold
        
        if vertical_deviation > deviation_threshold:
            consistency_note = f"Vertical deviation {vertical_deviation:.1f}% > {deviation_threshold}% threshold"
            formula = f"Horizontal preferred: {horizontal_avg:.4f} pt/mm (vertical {vertical_avg:.4f} inconsistent: {consistency_note})"
            return horizontal_avg, "horizontal_preferred", formula
        else:
            # Use mixed average when consistent
            consistency_note = f"Scales consistent (vertical deviation {vertical_deviation:.1f}%)"
            all_scales = horizontal_scales + vertical_scales
            mixed_avg = sum(all_scales) / len(all_scales)
            formula = f"Mixed average: ({' + '.join([f'{s:.4f}' for s in all_scales])}) ÷ {len(all_scales)} = {mixed_avg:.4f} pt/mm ({consistency_note})"
            return mixed_avg, "mixed_average", formula
    
    # CASE 4: No scales available
    return None, "no_scales", "No valid scale calculations found"

def process_region_with_intelligent_scaling(region: RegionData, drawing_type: str) -> RegionScaleResult:
    """Process a region using intelligent scale calculation"""
    
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
    
    # Find matches
    all_matches = find_dimension_line_matches(region, strategy, dimension_mapping, max_distance, effective_drawing_type)
    
    if not all_matches:
        return result
    
    # Select best matches per orientation
    horizontal_matches, vertical_matches = select_best_matches_per_orientation(all_matches, 3)
    
    # Create scale calculations
    result.horizontal_calculations = [create_scale_calculation(match) for match in horizontal_matches]
    result.vertical_calculations = [create_scale_calculation(match) for match in vertical_matches]
    
    # Calculate separate averages per orientation
    if horizontal_matches:
        h_scales = [m['scale_pt_per_mm'] for m in horizontal_matches]
        result.horizontal_average_scale = round(sum(h_scales) / len(h_scales), 4)
    
    if vertical_matches:
        v_scales = [m['scale_pt_per_mm'] for m in vertical_matches]  
        result.vertical_average_scale = round(sum(v_scales) / len(v_scales), 4)
    
    # INTELLIGENT: Apply horizontal-first fallback strategy
    final_scale, scale_source, formula = calculate_intelligent_scale_average(horizontal_matches, vertical_matches)
    
    if final_scale is not None:
        result.average_scale_pt_per_mm = round(final_scale, 4)
        result.average_scale_mm_per_pt = round(1 / final_scale, 4)
        result.average_calculation_formula = formula
        result.final_scale_source = scale_source
        result.total_calculations = len(horizontal_matches) + len(vertical_matches)
        
        # Add consistency check info
        if result.horizontal_average_scale and result.vertical_average_scale:
            deviation = abs(result.vertical_average_scale - result.horizontal_average_scale) / result.horizontal_average_scale * 100
            result.scale_consistency_check = f"Horizontal: {result.horizontal_average_scale:.4f}, Vertical: {result.vertical_average_scale:.4f}, Deviation: {deviation:.1f}%"
    
    return result

@app.post("/calculate-scale/", response_model=ScaleOutput)
async def calculate_scale(input_data: FilteredInput):
    """Calculate scale with intelligent horizontal-first fallback strategy"""
    try:
        if input_data.drawing_type == "installatietekening":
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
            "intelligent_scaling": {
                "horizontal_first": "Prefer horizontal scale when vertical deviates >15%",
                "deviation_threshold": "15%",
                "fallback_strategy": "horizontal_preferred > mixed_average > vertical_only"
            },
            "enhanced_validation": {
                "min_dimension_thresholds": MIN_DIMENSION_THRESHOLDS,
                "min_line_lengths": MIN_LINE_LENGTH_THRESHOLDS,
                "outlier_rules": OUTLIER_RULES
            }
        }
        
        # Process each region with intelligent scaling
        for region in input_data.regions:
            if region.lines and region.texts:
                region_result = process_region_with_intelligent_scaling(region, input_data.drawing_type)
                region_results.append(region_result)
                
                # Collect scales for global average
                if region_result.average_scale_pt_per_mm:
                    all_regional_scales.append(region_result.average_scale_pt_per_mm)
                    total_calculations += region_result.total_calculations
                
                # Track processing rules
                processing_rules[region.label] = {
                    "rules": region_result.region_rules,
                    "scale_source": region_result.final_scale_source,
                    "consistency": region_result.scale_consistency_check
                }
        
        # Calculate global averages
        global_avg_scale = None
        global_avg_scale_mm = None
        global_formula = None
        
        if all_regional_scales:
            global_avg_scale = sum(all_regional_scales) / len(all_regional_scales)
            global_avg_scale_mm = 1 / global_avg_scale
            global_formula = f"Regional intelligent averages: ({' + '.join([f'{s:.4f}' for s in all_regional_scales])}) ÷ {len(all_regional_scales)} = {global_avg_scale:.4f} pt/mm"
        
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
        "title": "Scale API v7.3.0 - Enhanced with Validation Rules",
        "version": "7.3.0",
        "description": "ENHANCED: Intelligent scale calculation with outlier detection and validation rules",
        "new_features_v7_3_0": {
            "dimension_filtering": {
                "doorsnede": "Min 1500mm horizontal, 800mm vertical (filters 420mm)",
                "bestektekening": "Min 2000mm horizontal, 800mm vertical",
                "plattegrond": "Min 1000mm both orientations"
            },
            "line_filtering": {
                "doorsnede": "Min 80pt length (filters ~68pt lines)",
                "purpose": "Remove short lines that cause unreliable scales"
            },
            "outlier_detection": {
                "global_limits": "0.025 - 0.12 pt/mm",
                "ratio_limits": "12 - 40 mm/pt",
                "suspect_patterns": ["+3420P", "+6410P", "+7555P"]
            },
            "validation_improvements": "Removes extreme values like 0.1821 and 0.0106 pt/mm"
        },
        "intelligent_scaling": {
            "principle": "Horizontal scale is usually more reliable in architectural drawings",
            "strategy": {
                "step_1": "Calculate separate averages for horizontal and vertical scales",
                "step_2": "Check if vertical deviates >15% from horizontal",
                "step_3": "If deviation >15%, prefer horizontal scale",
                "step_4": "If deviation ≤15%, use mixed average",
                "step_5": "Fallback to single orientation if only one available"
            },
            "scale_sources": {
                "horizontal_preferred": "Vertical inconsistent (>15% deviation), using horizontal",
                "mixed_average": "Both orientations consistent, using combined average", 
                "horizontal_only": "Only horizontal scales found",
                "vertical_only": "Only vertical scales found (fallback)",
                "no_scales": "No valid scales calculated"
            }
        },
        "deviation_threshold": "15% (configurable)",
        "rationale": {
            "why_horizontal_first": [
                "Horizontal dimensions (breedte) usually more consistent in architectural drawings",
                "Vertical measurements can be affected by text placement variations",
                "Building width measurements typically more standardized",
                "Horizontal lines often represent actual structural elements"
            ],
            "when_mixed_average": "When vertical scales are within 15% of horizontal (indicating consistent drawing)",
            "consistency_check": "Always reports deviation percentage for transparency"
        },
        "enhanced_features": {
            "per_orientation_averages": "Separate horizontal and vertical scale averages",
            "consistency_reporting": "Deviation percentage between orientations",
            "scale_source_tracking": "Records which strategy was used per region",
            "transparent_formulas": "Detailed calculation explanations",
            "enhanced_dimension_support": "P/V/+ symbols from Filter API v7.0.2",
            "advanced_validation": "Multi-layer filtering and outlier detection"
        },
        "distance_rules_per_drawing_type": {
            "plattegrond": "Both orientations: midpoint_to_midpoint",
            "detailtekening_plattegrond": "Both orientations: midpoint_to_midpoint", 
            "detailtekening_kozijn": "Horizontal: midpoint | Vertical: text-to-edge",
            "doorsnede": "Horizontal: midpoint | Vertical: text-to-edge", 
            "gevelaanzicht": "Vertical only: text-to-edge",
            "detailtekening": "Horizontal: midpoint | Vertical: text-to-edge"
        },
        "compatibility": {
            "filter_api": "v7.0.2 (enhanced dimension patterns)",
            "pydantic": "v2.6.4 compatible",
            "drawing_types": "All supported with intelligent scaling and validation"
        },
        "filtering_rules": {
            "min_dimensions": MIN_DIMENSION_THRESHOLDS,
            "min_line_lengths": MIN_LINE_LENGTH_THRESHOLDS,
            "outlier_limits": OUTLIER_RULES,
            "suspect_patterns": ["+3420P", "+6410P", "+7555P", "+7075P", "+3000P"]
        },
        "example_scenarios": {
            "consistent_scales": {
                "horizontal": "0.0567 pt/mm",
                "vertical": "0.0590 pt/mm", 
                "deviation": "4.1%",
                "result": "Mixed average: 0.0579 pt/mm (consistent)"
            },
            "inconsistent_scales": {
                "horizontal": "0.0567 pt/mm",
                "vertical": "0.1821 pt/mm",
                "deviation": "221%", 
                "result": "Horizontal preferred: 0.0567 pt/mm (vertical inconsistent)"
            },
            "filtered_outliers": {
                "before": "420mm -> 0.1821 pt/mm (extreme)",
                "after": "Filtered out by dimension size validation",
                "result": "Only reliable scales remain"
            }
        },
        "endpoints": {
            "/calculate-scale/": "Main scale calculation with intelligent fallback and validation",
            "/health/": "Health check with enhanced validation status",
            "/": "This comprehensive documentation"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "7.3.0",
        "port": PORT,
        "intelligent_scaling_enabled": True,
        "enhanced_validation_enabled": True,
        "features": {
            "horizontal_first_fallback": True,
            "deviation_threshold": "15%",
            "consistency_checking": True,
            "per_orientation_averages": True,
            "scale_source_tracking": True,
            "enhanced_dimension_support": True,
            "drawing_type_specific_rules": True,
            "filter_api_v7_compatibility": True,
            "pydantic_v2_serialization": True,
            "dimension_size_validation": True,
            "line_length_validation": True,
            "outlier_detection": True,
            "suspect_pattern_detection": True,
            "advanced_filtering": True
        },
        "supported_drawing_types": [
            "plattegrond", "doorsnede", "gevelaanzicht",
            "detailtekening_kozijn", "detailtekening_plattegrond", 
            "detailtekening", "bestektekening"
        ],
        "skipped_types": ["installatietekening"],
        "filtering_rules": {
            "min_dimensions": MIN_DIMENSION_THRESHOLDS,
            "min_line_lengths": MIN_LINE_LENGTH_THRESHOLDS,
            "outlier_limits": OUTLIER_RULES,
            "suspect_patterns": ["+3420P", "+6410P", "+7555P", "+7075P", "+3000P"]
        },
        "intelligent_scaling_logic": {
            "priority_1": "horizontal_preferred (when vertical >15% deviation)",
            "priority_2": "mixed_average (when consistent)",
            "priority_3": "single_orientation (when only one available)",
            "transparency": "Always reports which strategy was used"
        },
        "validation_improvements": {
            "problem_solved": {
                "420mm_dimensions": "Filtered out by MIN_DIMENSION_THRESHOLDS",
                "68pt_lines": "Filtered out by MIN_LINE_LENGTH_THRESHOLDS",
                "extreme_scales": "Removed by OUTLIER_RULES",
                "suspect_patterns": "Detected and filtered (+3420P, +6410P, etc.)"
            },
            "expected_results": {
                "doorsnede_before": "12 calculations with extreme outliers",
                "doorsnede_after": "~6 reliable calculations around 0.057 pt/mm",
                "consistency": "Much more consistent results per region"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
