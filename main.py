import os
import re
import math
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Scale Calculation API - Simplified",
    description="Calculates scale (px/mm) per region using highest dimensions only",
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
class RegionScaleResult(BaseModel):
    region_id: str
    horizontal: Optional[Dict[str, Any]] = None
    vertical: Optional[Dict[str, Any]] = None

class ScaleOutput(BaseModel):
    drawing_type: str
    regions: List[RegionScaleResult]
    timestamp: str

# Utility functions
def calculate_distance(p1: Dict[str, float], p2: CleanPoint) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1["x"])**2 + (p2.y - p1["y"])**2)

def extract_dimension_mm(text: str) -> Optional[float]:
    """Extract dimension value and convert to mm"""
    text = text.strip()
    
    # Pattern for valid dimensions: numbers with optional units
    pattern = r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$'
    match = re.match(pattern, text)
    
    if not match:
        return None
    
    try:
        value_str = match.group(1).replace(',', '.')
        value = float(value_str)
        unit = match.group(2) if match.group(2) else 'mm'
        
        # Convert to mm
        conversions = {
            'mm': 1.0,
            'cm': 10.0,
            'm': 1000.0
        }
        value_mm = value * conversions.get(unit, 1.0)
        
        return value_mm
        
    except (ValueError, IndexError):
        return None

def find_highest_dimension(texts: List[FilteredText], orientation: str) -> Optional[FilteredText]:
    """Find the text with highest dimension value for given orientation"""
    highest_value = 0
    highest_text = None
    
    for text in texts:
        if text.orientation != orientation:
            continue
        
        dim_value = extract_dimension_mm(text.text)
        if dim_value and dim_value > highest_value:
            highest_value = dim_value
            highest_text = text
    
    return highest_text

def find_closest_line(
    text_midpoint: Dict[str, float],
    lines: List[FilteredLine],
    orientation: str
) -> Optional[FilteredLine]:
    """Find the closest line with same orientation"""
    closest_line = None
    min_distance = float('inf')
    
    for line in lines:
        if line.orientation != orientation:
            continue
        
        distance = calculate_distance(text_midpoint, line.midpoint)
        
        if distance < min_distance:
            min_distance = distance
            closest_line = line
    
    return closest_line

def process_region(region: RegionData) -> RegionScaleResult:
    """Process a single region - find highest dimensions and calculate scales"""
    logger.info(f"Processing region: {region.label}")
    
    result = RegionScaleResult(region_id=region.label)
    
    # Process horizontal dimension
    horizontal_text = find_highest_dimension(region.texts, "horizontal")
    if horizontal_text:
        horizontal_line = find_closest_line(horizontal_text.midpoint, region.lines, "horizontal")
        if horizontal_line:
            dimension_mm = extract_dimension_mm(horizontal_text.text)
            if dimension_mm:
                scale_px_per_mm = horizontal_line.length / dimension_mm
                
                result.horizontal = {
                    "orientation": "horizontal",
                    "dimension_text": horizontal_text.text,
                    "dimension_mm": dimension_mm,
                    "line_length_px": horizontal_line.length,
                    "scale_px_per_mm": round(scale_px_per_mm, 4),
                    "distance_to_line": round(calculate_distance(horizontal_text.midpoint, horizontal_line.midpoint), 2)
                }
                
                logger.info(f"  Horizontal: {horizontal_text.text} = {dimension_mm}mm, line = {horizontal_line.length}px, scale = {scale_px_per_mm:.4f} px/mm")
    
    # Process vertical dimension
    vertical_text = find_highest_dimension(region.texts, "vertical")
    if vertical_text:
        vertical_line = find_closest_line(vertical_text.midpoint, region.lines, "vertical")
        if vertical_line:
            dimension_mm = extract_dimension_mm(vertical_text.text)
            if dimension_mm:
                scale_px_per_mm = vertical_line.length / dimension_mm
                
                result.vertical = {
                    "orientation": "vertical",
                    "dimension_text": vertical_text.text,
                    "dimension_mm": dimension_mm,
                    "line_length_px": vertical_line.length,
                    "scale_px_per_mm": round(scale_px_per_mm, 4),
                    "distance_to_line": round(calculate_distance(vertical_text.midpoint, vertical_line.midpoint), 2)
                }
                
                logger.info(f"  Vertical: {vertical_text.text} = {dimension_mm}mm, line = {vertical_line.length}px, scale = {scale_px_per_mm:.4f} px/mm")
    
    return result

@app.post("/calculate-scale/", response_model=ScaleOutput)
async def calculate_scale(input_data: FilteredInput):
    """Calculate scale for each region using highest dimensions only"""
    try:
        logger.info(f"=== Scale Calculation Start ===")
        logger.info(f"Processing {input_data.drawing_type} with {len(input_data.regions)} regions")
        
        region_results = []
        
        # Process each region
        for region in input_data.regions:
            logger.info(f"\nRegion {region.label}: {len(region.lines)} lines, {len(region.texts)} texts")
            
            if not region.lines or not region.texts:
                logger.warning(f"Skipping region {region.label} - no lines or texts")
                continue
            
            region_result = process_region(region)
            region_results.append(region_result)
        
        # Build response
        response = ScaleOutput(
            drawing_type=input_data.drawing_type,
            regions=region_results,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"\n=== Scale Calculation Complete ===")
        logger.info(f"Processed {len(region_results)} regions")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during scale calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Scale Calculation API - Simplified",
        "version": "3.0.0",
        "description": "Calculates scale (px/mm) per region using highest dimensions only",
        "workflow": [
            "1. For each region, find highest horizontal dimension text",
            "2. Find closest horizontal line to that text",
            "3. Calculate scale: line_length_px / dimension_mm",
            "4. Repeat for vertical dimension",
            "5. Return raw calculations only"
        ],
        "features": [
            "✅ Only highest dimension per orientation",
            "✅ Only closest line per dimension",
            "✅ Raw scale calculations only",
            "❌ No confidence scores",
            "❌ No fallback strategies",
            "❌ No weighted scoring"
        ],
        "output_format": {
            "horizontal": {
                "orientation": "horizontal",
                "dimension_text": "2000 mm",
                "dimension_mm": 2000.0,
                "line_length_px": 530.0,
                "scale_px_per_mm": 0.265,
                "distance_to_line": 12.5
            },
            "vertical": "same format for vertical orientation"
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
    logger.info(f"Starting Simplified Scale Calculation API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
