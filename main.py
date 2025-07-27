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
    title="Scale API v4.0.0 - Filter API Compatible",
    description="Calculates scale per drawing type/region: 3H + 3V = 6 calculations per region, max 12pt distance",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Models (Compatible with Filter API v6.0.0)
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
    
    # Calculation details
    calculation_formula: str  # "124.5pt ÷ 3000mm = 0.0415 pt/mm"
    scale_pt_per_mm: float
    scale_mm_per_pt: float
    orientation: str

class RegionScaleResult(BaseModel):
    region_label: str
    drawing_type: str
    
    # Calculations per orientation
    horizontal_calculations: List[ScaleCalculation] = []
    vertical_calculations: List[ScaleCalculation] = []
    
    # Region averages
    total_calculations: int = 0
    average_scale_pt_per_mm: Optional[float] = None
    average_scale_mm_per_pt: Optional[float] = None
    average_calculation_formula: Optional[str] = None  # Shows how average was calculated

class ScaleOutput(BaseModel):
    drawing_type: str
    total_regions: int
    total_calculations: int
    regions: List[RegionScaleResult]
    
    # Global average across all regions
    global_average_scale_pt_per_mm: Optional[float] = None
    global_average_scale_mm_per_pt: Optional[float] = None
    global_average_formula: Optional[str] = None
    
    timestamp: str

# Utility functions
def calculate_midpoint_distance(text_midpoint: Dict[str, float], line_midpoint: CleanPoint) -> float:
    """Calculate Euclidean distance between text midpoint and line midpoint"""
    return math.sqrt(
        (text_midpoint["x"] - line_midpoint.x)**2 + 
        (text_midpoint["y"] - line_midpoint.y)**2
    )

def extract_dimension_info(text: str) -> Optional[Tuple[float, str, float]]:
    """Extract dimension value, unit and convert to mm. Returns (value, unit, value_mm)"""
    text = text.strip()
    
    # Match numbers with optional units: "3000", "3000mm", "3,5 m", "250 cm"
    pattern = r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$'
    match = re.match(pattern, text)
    
    if not match:
        return None
    
    try:
        value_str = match.group(1).replace(',', '.')
        value = float(value_str)
        unit = match.group(2) if match.group(2) else 'mm'
        
        # Convert to mm
        conversions = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
        value_mm = value * conversions.get(unit, 1.0)
        
        return (value, unit, value_mm)
        
    except (ValueError, IndexError):
        return None

def get_top_dimensions_by_orientation(texts: List[FilteredText], target_orientation: str, limit: int = 3) -> List[Tuple[FilteredText, float, str, float]]:
    """Get top N dimensions that could match the target orientation"""
    valid_dimensions = []
    
    for text in texts:
        dim_result = extract_dimension_info(text.text)
        if dim_result:
            value, unit, value_mm = dim_result
            # Store all valid dimensions - we'll match by closest line orientation later
            valid_dimensions.append((text, value, unit, value_mm))
    
    # Sort by value_mm (descending) to get largest dimensions first
    valid_dimensions.sort(key=lambda x: x[3], reverse=True)
    return valid_dimensions

def find_best_line_match(text_midpoint: Dict[str, float], lines: List[FilteredLine], target_orientation: str, max_distance: float = 12.0) -> Optional[FilteredLine]:
    """Find the best matching line for a dimension text"""
    best_line = None
    min_distance = float('inf')
    
    # First pass: try to find lines with matching orientation
    for line in lines:
        if line.orientation == target_orientation:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                best_line = line
    
    # If no matching orientation found within distance, try any orientation
    if best_line is None:
        for line in lines:
            distance = calculate_midpoint_distance(text_midpoint, line.midpoint)
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                best_line = line
    
    return best_line

def process_region_calculations(region: RegionData, drawing_type: str) -> RegionScaleResult:
    """Process a region and calculate scales for 3H + 3V dimensions"""
    result = RegionScaleResult(
        region_label=region.label,
        drawing_type=drawing_type
    )
    
    all_scale_values = []
    calculation_details = []
    
    # Get all valid dimensions sorted by size
    all_dimensions = []
    for text in region.texts:
        dim_result = extract_dimension_info(text.text)
        if dim_result:
            value, unit, value_mm = dim_result
            all_dimensions.append((text, value, unit, value_mm))
    
    # Sort by size (largest first)
    all_dimensions.sort(key=lambda x: x[3], reverse=True)
    
    # Try to find 3 horizontal and 3 vertical matches
    used_texts = set()
    used_lines = set()
    
    # Process horizontal matches
    horizontal_count = 0
    for text, value, unit, value_mm in all_dimensions:
        if horizontal_count >= 3 or id(text) in used_texts:
            continue
            
        # Find available horizontal lines
        available_horizontal_lines = [line for line in region.lines 
                                    if line.orientation == "horizontal" and id(line) not in used_lines]
        
        if available_horizontal_lines:
            best_line = find_best_line_match(text.midpoint, available_horizontal_lines, "horizontal")
            if best_line:
                distance = calculate_midpoint_distance(text.midpoint, best_line.midpoint)
                scale_pt_per_mm = best_line.length / value_mm
                scale_mm_per_pt = 1 / scale_pt_per_mm
                
                formula = f"{best_line.length:.1f}pt ÷ {value_mm:.0f}mm = {scale_pt_per_mm:.4f} pt/mm"
                
                calc = ScaleCalculation(
                    dimension_text=text.text,
                    dimension_value=value,
                    dimension_unit=unit,
                    dimension_mm=value_mm,
                    line_length_pt=best_line.length,
                    text_midpoint=text.midpoint,
                    line_midpoint={"x": best_line.midpoint.x, "y": best_line.midpoint.y},
                    distance_text_to_line=round(distance, 2),
                    calculation_formula=formula,
                    scale_pt_per_mm=round(scale_pt_per_mm, 4),
                    scale_mm_per_pt=round(scale_mm_per_pt, 4),
                    orientation="horizontal"
                )
                
                result.horizontal_calculations.append(calc)
                all_scale_values.append(scale_pt_per_mm)
                calculation_details.append(f"H{horizontal_count+1}: {formula}")
                
                used_texts.add(id(text))
                used_lines.add(id(best_line))
                horizontal_count += 1
    
    # Process vertical matches
    vertical_count = 0
    for text, value, unit, value_mm in all_dimensions:
        if vertical_count >= 3 or id(text) in used_texts:
            continue
            
        # Find available vertical lines
        available_vertical_lines = [line for line in region.lines 
                                  if line.orientation == "vertical" and id(line) not in used_lines]
        
        if available_vertical_lines:
            best_line = find_best_line_match(text.midpoint, available_vertical_lines, "vertical")
            if best_line:
                distance = calculate_midpoint_distance(text.midpoint, best_line.midpoint)
                scale_pt_per_mm = best_line.length / value_mm
                scale_mm_per_pt = 1 / scale_pt_per_mm
                
                formula = f"{best_line.length:.1f}pt ÷ {value_mm:.0f}mm = {scale_pt_per_mm:.4f} pt/mm"
                
                calc = ScaleCalculation(
                    dimension_text=text.text,
                    dimension_value=value,
                    dimension_unit=unit,
                    dimension_mm=value_mm,
                    line_length_pt=best_line.length,
                    text_midpoint=text.midpoint,
                    line_midpoint={"x": best_line.midpoint.x, "y": best_line.midpoint.y},
                    distance_text_to_line=round(distance, 2),
                    calculation_formula=formula,
                    scale_pt_per_mm=round(scale_pt_per_mm, 4),
                    scale_mm_per_pt=round(scale_mm_per_pt, 4),
                    orientation="vertical"
                )
                
                result.vertical_calculations.append(calc)
                all_scale_values.append(scale_pt_per_mm)
                calculation_details.append(f"V{vertical_count+1}: {formula}")
                
                used_texts.add(id(text))
                used_lines.add(id(best_line))
                vertical_count += 1
    
    # Calculate region averages
    result.total_calculations = len(all_scale_values)
    
    if all_scale_values:
        avg_scale = sum(all_scale_values) / len(all_scale_values)
        result.average_scale_pt_per_mm = round(avg_scale, 4)
        result.average_scale_mm_per_pt = round(1 / avg_scale, 4)
        
        # Build average calculation formula
        scale_sum = sum(all_scale_values)
        result.average_calculation_formula = f"({' + '.join([f'{s:.4f}' for s in all_scale_values])}) ÷ {len(all_scale_values)} = {avg_scale:.4f} pt/mm"
    
    return result

@app.post("/calculate-scale/", response_model=ScaleOutput)
async def calculate_scale(input_data: FilteredInput):
    """
    Calculate scale per drawing type/region: 3H + 3V = 6 calculations per region
    Maximum distance for matches: 12pt between text and line midpoints
    """
    try:
        region_results = []
        all_regional_scales = []
        total_calculations = 0
        
        # Process each region
        for region in input_data.regions:
            if region.lines and region.texts:
                region_result = process_region_calculations(region, input_data.drawing_type)
                region_results.append(region_result)
                
                # Collect scales for global average
                if region_result.average_scale_pt_per_mm:
                    all_regional_scales.append(region_result.average_scale_pt_per_mm)
                    total_calculations += region_result.total_calculations
        
        # Calculate global averages
        global_avg_scale = None
        global_avg_scale_mm = None
        global_formula = None
        
        if all_regional_scales:
            global_avg_scale = sum(all_regional_scales) / len(all_regional_scales)
            global_avg_scale_mm = 1 / global_avg_scale
            global_formula = f"({' + '.join([f'{s:.4f}' for s in all_regional_scales])}) ÷ {len(all_regional_scales)} = {global_avg_scale:.4f} pt/mm"
        
        # Build response
        response = ScaleOutput(
            drawing_type=input_data.drawing_type,
            total_regions=len(region_results),
            total_calculations=total_calculations,
            regions=region_results,
            global_average_scale_pt_per_mm=round(global_avg_scale, 4) if global_avg_scale else None,
            global_average_scale_mm_per_pt=round(global_avg_scale_mm, 4) if global_avg_scale_mm else None,
            global_average_formula=global_formula,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scale calculation error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Scale API v4.0.0 - Filter API Compatible",
        "version": "4.0.0",
        "description": "Calculates scale per drawing type/region: 3H + 3V = 6 calculations per region",
        "specifications": {
            "calculations_per_region": "3 horizontal + 3 vertical = 6 total",
            "max_distance": "12pt between text and line midpoints",
            "distance_calculation": "Euclidean distance between midpoints",
            "matching_strategy": "Best available line within max distance"
        },
        "example_calculation": {
            "formula": "line_length_pt ÷ dimension_mm = scale_pt_per_mm",
            "example": "124.5pt ÷ 3000mm = 0.0415 pt/mm"
        },
        "output_structure": {
            "per_region": {
                "horizontal_calculations": "Up to 3 horizontal dimension-line matches",
                "vertical_calculations": "Up to 3 vertical dimension-line matches", 
                "average_scale": "Average of all calculations in region",
                "average_formula": "Shows calculation details"
            },
            "global": {
                "average_of_regions": "Average scale across all regions",
                "total_calculations": "Sum of all individual calculations"
            }
        },
        "expected_scenario": {
            "6_regions_x_6_calculations": "36 total calculations",
            "regional_averages": "6 regional average scales",
            "global_average": "1 final average scale"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "port": PORT,
        "max_distance": "12pt",
        "calculations_per_region": "3H + 3V = 6"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
