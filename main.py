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
    title="Scale Calculation API - Pure Calculations",
    description="Calculates scale (pt/mm) per region - top 3 dimensions per orientation",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Models
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
    orientation: str

class RegionData(BaseModel):
    label: str
    lines: List[FilteredLine]
    texts: List[FilteredText]

class FilteredInput(BaseModel):
    drawing_type: str
    regions: List[RegionData]

# Output Models
class DimensionCalculation(BaseModel):
    dimension_text: str
    dimension_value: float
    dimension_unit: str
    dimension_mm: float
    line_length_pt: float
    line_midpoint: Dict[str, float]
    text_midpoint: Dict[str, float]
    distance_text_to_line: float
    scale_pt_per_mm: float
    scale_mm_per_pt: float

class RegionScaleResult(BaseModel):
    region_label: str
    horizontal: List[DimensionCalculation] = []
    vertical: List[DimensionCalculation] = []
    average_scale_pt_per_mm: Optional[float] = None
    average_scale_mm_per_pt: Optional[float] = None

class ScaleOutput(BaseModel):
    drawing_type: str
    regions: List[RegionScaleResult]
    timestamp: str

# Utility functions
def calculate_distance(p1: Dict[str, float], p2: CleanPoint) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1["x"])**2 + (p2.y - p1["y"])**2)

def extract_dimension_mm(text: str) -> Optional[Tuple[float, str, float]]:
    """Extract dimension value, unit and convert to mm. Returns (value, unit, value_mm)"""
    text = text.strip()
    
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
        
        if value_mm < 100:  # Skip very small dimensions
            return None
        
        return (value, unit, value_mm)
        
    except (ValueError, IndexError):
        return None

def find_top_dimensions(texts: List[FilteredText], orientation: str, limit: int = 3) -> List[Tuple[FilteredText, float, str, float]]:
    """Find top N dimension values for given orientation. Returns (text, value, unit, value_mm)"""
    valid_dimensions = []
    
    for text in texts:
        if text.orientation != orientation:
            continue
        
        dim_result = extract_dimension_mm(text.text)
        if dim_result:
            value, unit, value_mm = dim_result
            valid_dimensions.append((text, value, unit, value_mm))
    
    # Sort by value_mm (descending) and take top N
    valid_dimensions.sort(key=lambda x: x[3], reverse=True)
    return valid_dimensions[:limit]

def find_closest_line(text_midpoint: Dict[str, float], lines: List[FilteredLine], orientation: str) -> Optional[FilteredLine]:
    """Find the closest line with same orientation"""
    closest_line = None
    min_distance = float('inf')
    
    for line in lines:
        if line.orientation != orientation:
            continue
        
        if line.length < 50:  # Skip very short lines
            continue
        
        distance = calculate_distance(text_midpoint, line.midpoint)
        
        if distance < min_distance:
            min_distance = distance
            closest_line = line
    
    return closest_line

def process_region(region: RegionData) -> RegionScaleResult:
    """Process a single region - find top 3 dimensions per orientation and calculate scales"""
    result = RegionScaleResult(region_label=region.label)
    
    all_scales = []
    
    # Process horizontal dimensions (top 3)
    horizontal_dims = find_top_dimensions(region.texts, "horizontal", 3)
    for text, value, unit, value_mm in horizontal_dims:
        closest_line = find_closest_line(text.midpoint, region.lines, "horizontal")
        if closest_line:
            distance = calculate_distance(text.midpoint, closest_line.midpoint)
            scale_pt_per_mm = closest_line.length / value_mm
            scale_mm_per_pt = 1 / scale_pt_per_mm
            
            calc = DimensionCalculation(
                dimension_text=text.text,
                dimension_value=value,
                dimension_unit=unit,
                dimension_mm=value_mm,
                line_length_pt=closest_line.length,
                line_midpoint={"x": closest_line.midpoint.x, "y": closest_line.midpoint.y},
                text_midpoint=text.midpoint,
                distance_text_to_line=round(distance, 2),
                scale_pt_per_mm=round(scale_pt_per_mm, 4),
                scale_mm_per_pt=round(scale_mm_per_pt, 4)
            )
            result.horizontal.append(calc)
            all_scales.append(scale_pt_per_mm)
    
    # Process vertical dimensions (top 3)
    vertical_dims = find_top_dimensions(region.texts, "vertical", 3)
    for text, value, unit, value_mm in vertical_dims:
        closest_line = find_closest_line(text.midpoint, region.lines, "vertical")
        if closest_line:
            distance = calculate_distance(text.midpoint, closest_line.midpoint)
            scale_pt_per_mm = closest_line.length / value_mm
            scale_mm_per_pt = 1 / scale_pt_per_mm
            
            calc = DimensionCalculation(
                dimension_text=text.text,
                dimension_value=value,
                dimension_unit=unit,
                dimension_mm=value_mm,
                line_length_pt=closest_line.length,
                line_midpoint={"x": closest_line.midpoint.x, "y": closest_line.midpoint.y},
                text_midpoint=text.midpoint,
                distance_text_to_line=round(distance, 2),
                scale_pt_per_mm=round(scale_pt_per_mm, 4),
                scale_mm_per_pt=round(scale_mm_per_pt, 4)
            )
            result.vertical.append(calc)
            all_scales.append(scale_pt_per_mm)
    
    # Calculate average scale from all calculations
    if all_scales:
        avg_scale = sum(all_scales) / len(all_scales)
        result.average_scale_pt_per_mm = round(avg_scale, 4)
        result.average_scale_mm_per_pt = round(1 / avg_scale, 4)
    
    return result

@app.post("/calculate-scale/", response_model=ScaleOutput)
async def calculate_scale(input_data: FilteredInput):
    """Calculate scale for each region using top 3 dimensions per orientation"""
    region_results = []
    
    # Process each region
    for region in input_data.regions:
        if region.lines and region.texts:
            region_result = process_region(region)
            region_results.append(region_result)
    
    # Build response
    response = ScaleOutput(
        drawing_type=input_data.drawing_type,
        regions=region_results,
        timestamp=datetime.now().isoformat()
    )
    
    return response

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Scale Calculation API - Pure Calculations",
        "version": "3.0.0",
        "description": "Calculates scale (pt/mm) per region using top 3 dimensions per orientation",
        "workflow": [
            "1. For each region, finds top 3 horizontal and vertical dimensions",
            "2. For each dimension, finds closest line with same orientation", 
            "3. Calculates scale: line_length_pt / dimension_mm",
            "4. Returns detailed calculation info per dimension"
        ],
        "features": [
            "✅ Top 3 dimensions per orientation per region",
            "✅ Closest line matching per dimension",
            "✅ Pure scale calculations only",
            "❌ No logging, confidence, validation, or fallbacks"
        ],
        "output_format": {
            "horizontal": "Array of up to 3 dimension calculations",
            "vertical": "Array of up to 3 dimension calculations",
            "average": "Average scale from all calculations in region"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "port": PORT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
