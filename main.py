import os
import re
import math
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PORT = int(os.environ.get("PORT", 10000))
DISTANCE_THRESHOLD = 10.0  # Max distance in points for matching
SCALE_TOLERANCE = 0.1  # 10% tolerance for horizontal/vertical scale comparison

# Drawing type specific scale ranges (pt/mm)
SCALE_RANGES = {
    "plattegrond": {"min": 0.10, "max": 0.25},
    "gevelaanzicht": {"min": 0.15, "max": 0.35},
    "detailtekening": {"min": 0.20, "max": 1.00},
    "doorsnede": {"min": 0.15, "max": 0.35},
    "bestektekening": {"min": 0.10, "max": 0.30},
    "installatietekening": {"min": 0.15, "max": 0.60}
}

# Default scales per drawing type (pt/mm)
DEFAULT_SCALES = {
    "plattegrond": 0.15,
    "gevelaanzicht": 0.20,
    "detailtekening": 0.50,
    "doorsnede": 0.20,
    "bestektekening": 0.15,
    "installatietekening": 0.30
}

app = FastAPI(
    title="Scale Calculation API",
    description="Calculates scale (pt/mm) per region from filtered vector data",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Models - matching Filter API output format
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
class DimensionData(BaseModel):
    value: float
    unit: str
    value_mm: float
    text: str

class LineMatch(BaseModel):
    line: FilteredLine
    distance: float

class ScaleMatch(BaseModel):
    dimension: DimensionData
    line_match: LineMatch
    scale_pt_per_mm: float
    scale_mm_per_pt: float

class RegionScaleResult(BaseModel):
    region_label: str
    horizontal: Optional[ScaleMatch] = None
    vertical: Optional[ScaleMatch] = None
    scales_match: bool
    scale_ratio: Optional[float] = None
    confidence: float
    average_scale_pt_per_mm: Optional[float] = None
    average_scale_mm_per_pt: Optional[float] = None
    validation_status: str

class ScaleOutput(BaseModel):
    drawing_type: str
    regions: List[RegionScaleResult]
    overall_average_pt_per_mm: Optional[float] = None
    overall_average_mm_per_pt: Optional[float] = None
    timestamp: str

# Utility functions
def calculate_distance(p1: Dict[str, float], p2: CleanPoint) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1["x"])**2 + (p2.y - p1["y"])**2)

def extract_dimension(text: str) -> Optional[DimensionData]:
    """Extract dimension value and unit from text"""
    text = text.strip()
    
    # Pattern for valid dimensions: pure numbers with optional units
    pattern = r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$'
    match = re.match(pattern, text)
    
    if not match:
        return None
    
    try:
        value_str = match.group(1).replace(',', '.')
        value = float(value_str)
        unit = match.group(2) if match.group(2) else 'mm'
        
        # Skip very small values
        if value < 100 and unit == 'mm':
            return None
        
        # Convert to mm
        conversions = {
            'mm': 1.0,
            'cm': 10.0,
            'm': 1000.0
        }
        value_mm = value * conversions.get(unit, 1.0)
        
        # Skip dimensions less than 100mm
        if value_mm < 100:
            return None
        
        return DimensionData(
            value=value,
            unit=unit,
            value_mm=value_mm,
            text=text
        )
    except (ValueError, IndexError):
        return None

def find_highest_dimension(texts: List[FilteredText], orientation: str) -> Optional[Tuple[FilteredText, DimensionData]]:
    """Find the highest dimension value for a given orientation"""
    highest_value = 0
    highest_text = None
    highest_dim = None
    
    for text in texts:
        if text.orientation != orientation:
            continue
        
        dim = extract_dimension(text.text)
        if dim and dim.value_mm > highest_value:
            highest_value = dim.value_mm
            highest_text = text
            highest_dim = dim
    
    return (highest_text, highest_dim) if highest_text else None

def find_nearest_line(
    dimension_midpoint: Dict[str, float],
    lines: List[FilteredLine],
    orientation: str,
    threshold: Optional[float] = DISTANCE_THRESHOLD
) -> Optional[LineMatch]:
    """Find the nearest line with same orientation"""
    nearest_line = None
    min_distance = float('inf')
    
    for line in lines:
        if line.orientation != orientation:
            continue
        
        # Skip very short lines
        if line.length < 50:
            continue
        
        distance = calculate_distance(dimension_midpoint, line.midpoint)
        
        if threshold is None or distance <= threshold:
            if distance < min_distance:
                min_distance = distance
                nearest_line = line
    
    if nearest_line:
        return LineMatch(line=nearest_line, distance=min_distance)
    
    return None

def calculate_confidence(distance: float, line_length: float) -> float:
    """Calculate confidence score based on distance and line length"""
    # Distance component (0-1, lower distance = higher score)
    distance_score = max(0, 1 - (distance / DISTANCE_THRESHOLD))
    
    # Line length component (0-1, longer lines = higher score)
    length_score = min(1, line_length / 1000)
    
    # Weighted average
    confidence = 0.6 * distance_score + 0.4 * length_score
    
    return round(confidence * 100, 1)

def validate_scale(scale_pt_per_mm: float, drawing_type: str) -> str:
    """Validate scale against expected ranges"""
    expected_range = SCALE_RANGES.get(drawing_type, {"min": 0.1, "max": 2.0})
    
    if expected_range["min"] <= scale_pt_per_mm <= expected_range["max"]:
        return "valid"
    elif scale_pt_per_mm < expected_range["min"]:
        return f"too_low (expected: {expected_range['min']}-{expected_range['max']} pt/mm)"
    else:
        return f"too_high (expected: {expected_range['min']}-{expected_range['max']} pt/mm)"

def process_region(region: RegionData, drawing_type: str) -> RegionScaleResult:
    """Process a single region to calculate scales"""
    logger.info(f"Processing region: {region.label}")
    
    result = RegionScaleResult(
        region_label=region.label,
        scales_match=False,
        confidence=0.0,
        validation_status="no_data"
    )
    
    # Find highest horizontal dimension
    horizontal_result = find_highest_dimension(region.texts, "horizontal")
    if horizontal_result:
        h_text, h_dim = horizontal_result
        logger.info(f"  Highest horizontal dimension: {h_dim.text} = {h_dim.value_mm}mm")
        
        # Find nearest horizontal line
        h_match = find_nearest_line(h_text.midpoint, region.lines, "horizontal")
        if not h_match:
            # Fallback: find nearest without threshold
            h_match = find_nearest_line(h_text.midpoint, region.lines, "horizontal", threshold=None)
        
        if h_match:
            scale_mm_per_pt = h_dim.value_mm / h_match.line.length
            scale_pt_per_mm = 1 / scale_mm_per_pt
            
            result.horizontal = ScaleMatch(
                dimension=h_dim,
                line_match=h_match,
                scale_pt_per_mm=round(scale_pt_per_mm, 4),
                scale_mm_per_pt=round(scale_mm_per_pt, 4)
            )
            
            logger.info(f"    Matched to line: {h_match.line.length:.1f}pt, distance: {h_match.distance:.1f}pt")
            logger.info(f"    Scale: {scale_pt_per_mm:.4f} pt/mm")
    
    # Find highest vertical dimension
    vertical_result = find_highest_dimension(region.texts, "vertical")
    if vertical_result:
        v_text, v_dim = vertical_result
        logger.info(f"  Highest vertical dimension: {v_dim.text} = {v_dim.value_mm}mm")
        
        # Find nearest vertical line
        v_match = find_nearest_line(v_text.midpoint, region.lines, "vertical")
        if not v_match:
            # Fallback: find nearest without threshold
            v_match = find_nearest_line(v_text.midpoint, region.lines, "vertical", threshold=None)
        
        if v_match:
            scale_mm_per_pt = v_dim.value_mm / v_match.line.length
            scale_pt_per_mm = 1 / scale_mm_per_pt
            
            result.vertical = ScaleMatch(
                dimension=v_dim,
                line_match=v_match,
                scale_pt_per_mm=round(scale_pt_per_mm, 4),
                scale_mm_per_pt=round(scale_mm_per_pt, 4)
            )
            
            logger.info(f"    Matched to line: {v_match.line.length:.1f}pt, distance: {v_match.distance:.1f}pt")
            logger.info(f"    Scale: {scale_pt_per_mm:.4f} pt/mm")
    
    # Compare scales and calculate average
    if result.horizontal and result.vertical:
        h_scale = result.horizontal.scale_pt_per_mm
        v_scale = result.vertical.scale_pt_per_mm
        
        # Calculate ratio
        ratio = h_scale / v_scale if v_scale > 0 else 0
        result.scale_ratio = round(ratio, 3)
        
        # Check if scales match (within tolerance)
        result.scales_match = 0.9 <= ratio <= 1.1
        
        # Calculate average
        result.average_scale_pt_per_mm = round((h_scale + v_scale) / 2, 4)
        result.average_scale_mm_per_pt = round(1 / result.average_scale_pt_per_mm, 4)
        
        # Calculate confidence
        h_conf = calculate_confidence(
            result.horizontal.line_match.distance,
            result.horizontal.line_match.line.length
        )
        v_conf = calculate_confidence(
            result.vertical.line_match.distance,
            result.vertical.line_match.line.length
        )
        result.confidence = round((h_conf + v_conf) / 2, 1)
        
        # Validate scale
        result.validation_status = validate_scale(result.average_scale_pt_per_mm, drawing_type)
        
        logger.info(f"  Scale match: {result.scales_match} (ratio: {ratio:.3f})")
        logger.info(f"  Average scale: {result.average_scale_pt_per_mm:.4f} pt/mm")
        logger.info(f"  Confidence: {result.confidence}%")
        
    elif result.horizontal:
        # Only horizontal scale available
        result.average_scale_pt_per_mm = result.horizontal.scale_pt_per_mm
        result.average_scale_mm_per_pt = result.horizontal.scale_mm_per_pt
        result.confidence = calculate_confidence(
            result.horizontal.line_match.distance,
            result.horizontal.line_match.line.length
        )
        result.validation_status = validate_scale(result.average_scale_pt_per_mm, drawing_type)
        
    elif result.vertical:
        # Only vertical scale available
        result.average_scale_pt_per_mm = result.vertical.scale_pt_per_mm
        result.average_scale_mm_per_pt = result.vertical.scale_mm_per_pt
        result.confidence = calculate_confidence(
            result.vertical.line_match.distance,
            result.vertical.line_match.line.length
        )
        result.validation_status = validate_scale(result.average_scale_pt_per_mm, drawing_type)
    else:
        # No scales found - use default
        logger.warning(f"  No dimensions found, using default scale for {drawing_type}")
        default_scale = DEFAULT_SCALES.get(drawing_type, 0.35)
        result.average_scale_pt_per_mm = default_scale
        result.average_scale_mm_per_pt = round(1 / default_scale, 4)
        result.confidence = 0.0
        result.validation_status = "default_used"
    
    return result

@app.post("/calculate-scale/", response_model=ScaleOutput)
async def calculate_scale(input_data: FilteredInput, debug: bool = Query(False)):
    """Calculate scale for each region from filtered data"""
    try:
        logger.info(f"=== Scale Calculation Start ===")
        logger.info(f"Processing {input_data.drawing_type} with {len(input_data.regions)} regions")
        logger.info(f"Debug mode: {debug}")
        
        region_results = []
        valid_scales = []
        
        # Process each region
        for region in input_data.regions:
            logger.info(f"\nRegion {region.label}: {len(region.lines)} lines, {len(region.texts)} texts")
            
            if not region.lines or not region.texts:
                logger.warning(f"Skipping region {region.label} - no lines or texts")
                continue
            
            region_result = process_region(region, input_data.drawing_type)
            region_results.append(region_result)
            
            # Collect valid scales for overall average
            if region_result.average_scale_pt_per_mm and region_result.validation_status == "valid":
                valid_scales.append(region_result.average_scale_pt_per_mm)
        
        # Calculate overall average
        if valid_scales:
            overall_avg_pt_per_mm = round(np.mean(valid_scales), 4)
            overall_avg_mm_per_pt = round(1 / overall_avg_pt_per_mm, 4)
        else:
            # Use default for drawing type
            default_scale = DEFAULT_SCALES.get(input_data.drawing_type, 0.35)
            overall_avg_pt_per_mm = default_scale
            overall_avg_mm_per_pt = round(1 / default_scale, 4)
            logger.warning(f"No valid scales found, using default: {default_scale} pt/mm")
        
        # Build response
        response = ScaleOutput(
            drawing_type=input_data.drawing_type,
            regions=region_results,
            overall_average_pt_per_mm=overall_avg_pt_per_mm,
            overall_average_mm_per_pt=overall_avg_mm_per_pt,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"\n=== Scale Calculation Complete ===")
        logger.info(f"Overall average: {overall_avg_pt_per_mm:.4f} pt/mm ({overall_avg_mm_per_pt:.4f} mm/pt)")
        logger.info(f"Valid regions: {len([r for r in region_results if r.validation_status == 'valid'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during scale calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Scale Calculation API",
        "version": "2.0.0",
        "description": "Calculates scale (pt/mm) per region from filtered vector data",
        "endpoints": {
            "/": "This page",
            "/calculate-scale/": "POST - Calculate scale for each region",
            "/health/": "GET - Health check"
        },
        "workflow": [
            "1. Receives filtered data with regions containing lines and dimension texts",
            "2. For each region, finds highest horizontal and vertical dimension",
            "3. Matches each dimension to nearest line with same orientation",
            "4. Calculates scale by dividing dimension (mm) by line length (pt)",
            "5. Validates scales against drawing type ranges",
            "6. Returns scale per region with confidence scores"
        ],
        "features": [
            "Deterministic: Always selects highest dimension value",
            "10pt distance threshold for matching (with fallback)",
            "Separate horizontal/vertical scales with comparison",
            "Confidence scoring based on distance and line length",
            "Drawing type specific validation ranges",
            "Default scales for edge cases"
        ],
        "scale_ranges": SCALE_RANGES,
        "default_scales": DEFAULT_SCALES
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "port": PORT
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Scale Calculation API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
