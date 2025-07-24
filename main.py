import os
import re
import math
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException
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
DISTANCE_THRESHOLD_FACTOR = 0.1  # 10% of average line length for regions
MIN_DISTANCE_THRESHOLD = 20.0
OUTLIER_Z_SCORE = 2.0

app = FastAPI(
    title="Region-Based Scale Calculation API",
    description="Calculates scale per region from filtered vector data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Models - matching Filter API output
class CleanPoint(BaseModel):
    x: float
    y: float

class CleanLine(BaseModel):
    p1: CleanPoint
    p2: CleanPoint
    length: float
    orientation: str
    midpoint: CleanPoint

class CleanText(BaseModel):
    text: str
    position: CleanPoint
    bounding_box: List[float]

class RegionData(BaseModel):
    label: str
    lines: List[CleanLine]
    texts: List[CleanText]

class FilteredInput(BaseModel):
    drawing_type: str
    regions: List[RegionData]

# Output Models
class DimensionData(BaseModel):
    value: float
    unit: str
    value_mm: float

class MatchedLine(BaseModel):
    p1: Dict[str, float]
    p2: Dict[str, float]
    length: float
    orientation: str

class MatchedText(BaseModel):
    value: str
    position: Dict[str, float]

class DimensionMatch(BaseModel):
    line: MatchedLine
    text: MatchedText
    dimension: DimensionData
    match_distance: float
    points_per_mm: float

class ScaleResult(BaseModel):
    points_per_cm: float
    cm_per_point: float
    points_per_m: float
    m_per_point: float
    confidence: float
    samples: int

class RegionScaleResult(BaseModel):
    region_label: str
    scale: Dict[str, ScaleResult]  # horizontal/vertical
    average: ScaleResult
    valid_pairs_used: int
    total_matches_checked: int
    matched_dimensions: List[DimensionMatch]

class ScaleOutput(BaseModel):
    drawing_type: str
    regions: List[RegionScaleResult]
    overall_average: ScaleResult
    total_valid_pairs: int

# Utility functions
def calculate_distance(p1: CleanPoint, p2: CleanPoint) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def calculate_perpendicular_distance(line_p1: CleanPoint, line_p2: CleanPoint, point: CleanPoint) -> float:
    """Calculate perpendicular distance from point to line segment"""
    dx = line_p2.x - line_p1.x
    dy = line_p2.y - line_p1.y
    px = point.x - line_p1.x
    py = point.y - line_p1.y
    line_length_sq = dx * dx + dy * dy
    
    if line_length_sq == 0:
        return math.sqrt(px * px + py * py)
    
    t = max(0, min(1, (px * dx + py * dy) / line_length_sq))
    proj_x = line_p1.x + t * dx
    proj_y = line_p1.y + t * dy
    
    return math.sqrt((point.x - proj_x)**2 + (point.y - proj_y)**2)

def extract_dimension(text: str) -> Optional[DimensionData]:
    """Extract dimension value and unit from text"""
    text = text.strip()
    
    # Common dimension patterns
    patterns = [
        (r'(\d+(?:[.,]\d+)?)\s*(mm|cm|m|meter|metre)\b', 'metric'),
        (r'(\d+(?:[.,]\d+)?)\s*(?:"|″|inch|inches|in)\b', 'inch'),
        (r'(\d+(?:[.,]\d+)?)\s*(?:\'|′|ft|foot|feet)\b', 'feet'),
        (r'^(\d{3,5})$', 'mm'),  # 3-5 digit numbers assumed to be mm
        (r'^(\d+)$', 'mm'),  # Any standalone number assumed to be mm
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                # Skip very small values
                if value < 10:
                    continue
                
                if pattern_type == 'metric' and len(match.groups()) > 1:
                    unit = match.group(2).lower()
                elif pattern_type == 'inch':
                    unit = 'in'
                elif pattern_type == 'feet':
                    unit = 'ft'
                else:
                    unit = 'mm'
                
                # Convert to mm
                conversions = {
                    'mm': 1.0,
                    'cm': 10.0,
                    'm': 1000.0,
                    'meter': 1000.0,
                    'metre': 1000.0,
                    'in': 25.4,
                    'ft': 304.8
                }
                value_mm = value * conversions.get(unit, 1.0)
                
                return DimensionData(value=value, unit=unit, value_mm=value_mm)
            except (ValueError, IndexError):
                continue
    
    return None

def match_dimensions_in_region(region: RegionData, orientation: str) -> Tuple[List[Dict], int]:
    """Match dimension texts to lines within a region"""
    matches = []
    total_checked = 0
    
    # Filter lines by orientation
    oriented_lines = [line for line in region.lines if line.orientation == orientation]
    
    if not oriented_lines:
        return [], 0
    
    # Calculate average line length for threshold
    avg_line_length = np.mean([line.length for line in oriented_lines])
    threshold = max(MIN_DISTANCE_THRESHOLD, avg_line_length * DISTANCE_THRESHOLD_FACTOR)
    
    logger.info(f"Region {region.label} - {orientation}: {len(oriented_lines)} lines, threshold: {threshold:.1f}")
    
    for line in oriented_lines:
        best_match = None
        best_distance = float('inf')
        
        for text in region.texts:
            total_checked += 1
            
            # Calculate distances
            perp_dist = calculate_perpendicular_distance(line.p1, line.p2, text.position)
            mid_dist = calculate_distance(line.midpoint, text.position)
            min_dist = min(perp_dist, mid_dist)
            
            if min_dist < threshold and min_dist < best_distance:
                dim = extract_dimension(text.text)
                if dim and dim.value_mm > 0:
                    best_distance = min_dist
                    best_match = (text, dim)
        
        if best_match:
            text, dim = best_match
            points_per_mm = line.length / dim.value_mm
            
            matches.append({
                "line": line,
                "text": text,
                "dimension": dim,
                "match_distance": best_distance,
                "points_per_mm": points_per_mm
            })
            
            logger.info(f"  Match: {dim.value}{dim.unit} -> {line.length:.1f}pt = {points_per_mm:.3f} pt/mm")
    
    return matches, total_checked

def filter_outliers(matches: List[Dict]) -> List[Dict]:
    """Filter outliers using z-score method"""
    if len(matches) < 3:
        return matches
    
    scales = [m['points_per_mm'] for m in matches]
    mean_scale = np.mean(scales)
    std_scale = np.std(scales)
    
    if std_scale == 0:
        return matches
    
    z_scores = [(s - mean_scale) / std_scale for s in scales]
    filtered_matches = [m for i, m in enumerate(matches) if abs(z_scores[i]) <= OUTLIER_Z_SCORE]
    
    # Keep at least 30% of matches
    if len(filtered_matches) < max(1, len(matches) * 0.3):
        sorted_indices = sorted(range(len(matches)), key=lambda i: abs(z_scores[i]))
        keep_count = max(1, int(len(matches) * 0.3))
        filtered_matches = [matches[i] for i in sorted_indices[:keep_count]]
    
    return filtered_matches

def calculate_scale_stats(matches: List[Dict]) -> ScaleResult:
    """Calculate scale statistics from matches"""
    if not matches:
        return ScaleResult(
            points_per_cm=0.0,
            cm_per_point=0.0,
            points_per_m=0.0,
            m_per_point=0.0,
            confidence=0.0,
            samples=0
        )
    
    scales_mm = [m['points_per_mm'] for m in matches]
    mean_scale_mm = np.mean(scales_mm)
    
    # Calculate confidence based on consistency
    if len(scales_mm) > 1:
        std_scale_mm = np.std(scales_mm)
        consistency = std_scale_mm / mean_scale_mm if mean_scale_mm > 0 else 1.0
        confidence = max(0, min(100, 100 * (1 - consistency)))
    else:
        confidence = 50.0  # Single sample
    
    # Adjust confidence based on sample size
    sample_factor = min(1.0, len(matches) / 5.0)
    confidence = confidence * sample_factor
    
    points_per_cm = mean_scale_mm * 10
    points_per_m = mean_scale_mm * 1000
    
    return ScaleResult(
        points_per_cm=round(points_per_cm, 4),
        cm_per_point=round(1 / points_per_cm, 4) if points_per_cm > 0 else 0.0,
        points_per_m=round(points_per_m, 4),
        m_per_point=round(1 / points_per_m, 6) if points_per_m > 0 else 0.0,
        confidence=round(confidence, 1),
        samples=len(matches)
    )

def calculate_region_scale(region: RegionData) -> RegionScaleResult:
    """Calculate scale for a single region"""
    logger.info(f"Processing region: {region.label}")
    
    # Match dimensions for horizontal lines
    horiz_matches, horiz_checked = match_dimensions_in_region(region, "horizontal")
    horiz_filtered = filter_outliers(horiz_matches)
    
    # Match dimensions for vertical lines
    vert_matches, vert_checked = match_dimensions_in_region(region, "vertical")
    vert_filtered = filter_outliers(vert_matches)
    
    # Calculate scale statistics
    horizontal_scale = calculate_scale_stats(horiz_filtered)
    vertical_scale = calculate_scale_stats(vert_filtered)
    
    # Calculate average scale
    all_filtered = horiz_filtered + vert_filtered
    
    if all_filtered:
        # Weighted average based on sample count
        total_samples = horizontal_scale.samples + vertical_scale.samples
        if total_samples > 0:
            horiz_weight = horizontal_scale.samples / total_samples
            vert_weight = vertical_scale.samples / total_samples
            
            avg_points_per_cm = (
                horizontal_scale.points_per_cm * horiz_weight +
                vertical_scale.points_per_cm * vert_weight
            )
            avg_confidence = (
                horizontal_scale.confidence * horiz_weight +
                vertical_scale.confidence * vert_weight
            )
            
            average_scale = ScaleResult(
                points_per_cm=round(avg_points_per_cm, 4),
                cm_per_point=round(1 / avg_points_per_cm, 4) if avg_points_per_cm > 0 else 0.0,
                points_per_m=round(avg_points_per_cm * 100, 4),
                m_per_point=round(1 / (avg_points_per_cm * 100), 6) if avg_points_per_cm > 0 else 0.0,
                confidence=round(avg_confidence, 1),
                samples=total_samples
            )
        else:
            average_scale = calculate_scale_stats(all_filtered)
    else:
        average_scale = ScaleResult(
            points_per_cm=0.0,
            cm_per_point=0.0,
            points_per_m=0.0,
            m_per_point=0.0,
            confidence=0.0,
            samples=0
        )
    
    # Prepare matched dimensions for output
    matched_dimensions = []
    for m in all_filtered:
        matched_dimensions.append(DimensionMatch(
            line=MatchedLine(
                p1={"x": m['line'].p1.x, "y": m['line'].p1.y},
                p2={"x": m['line'].p2.x, "y": m['line'].p2.y},
                length=m['line'].length,
                orientation=m['line'].orientation
            ),
            text=MatchedText(
                value=m['text'].text,
                position={"x": m['text'].position.x, "y": m['text'].position.y}
            ),
            dimension=m['dimension'],
            match_distance=round(m['match_distance'], 2),
            points_per_mm=round(m['points_per_mm'], 4)
        ))
    
    # Build scale dictionary
    scale_dict = {}
    if horizontal_scale.samples > 0:
        scale_dict["horizontal"] = horizontal_scale
    if vertical_scale.samples > 0:
        scale_dict["vertical"] = vertical_scale
    
    return RegionScaleResult(
        region_label=region.label,
        scale=scale_dict,
        average=average_scale,
        valid_pairs_used=len(all_filtered),
        total_matches_checked=horiz_checked + vert_checked,
        matched_dimensions=matched_dimensions
    )

@app.post("/calculate-region-scales/", response_model=ScaleOutput)
async def calculate_region_scales(input_data: FilteredInput):
    """Calculate scale for each region from filtered data"""
    try:
        logger.info(f"Processing {input_data.drawing_type} with {len(input_data.regions)} regions")
        
        region_results = []
        all_valid_pairs = 0
        
        # Process each region
        for region in input_data.regions:
            logger.info(f"Region {region.label}: {len(region.lines)} lines, {len(region.texts)} texts")
            
            if not region.lines or not region.texts:
                logger.warning(f"Skipping region {region.label} - no lines or texts")
                continue
            
            region_result = calculate_region_scale(region)
            region_results.append(region_result)
            all_valid_pairs += region_result.valid_pairs_used
        
        # Calculate overall average
        if region_results:
            total_samples = sum(r.average.samples for r in region_results)
            if total_samples > 0:
                # Weighted average across all regions
                weighted_points_per_cm = sum(
                    r.average.points_per_cm * r.average.samples 
                    for r in region_results
                ) / total_samples
                
                weighted_confidence = sum(
                    r.average.confidence * r.average.samples 
                    for r in region_results
                ) / total_samples
                
                overall_average = ScaleResult(
                    points_per_cm=round(weighted_points_per_cm, 4),
                    cm_per_point=round(1 / weighted_points_per_cm, 4) if weighted_points_per_cm > 0 else 0.0,
                    points_per_m=round(weighted_points_per_cm * 100, 4),
                    m_per_point=round(1 / (weighted_points_per_cm * 100), 6) if weighted_points_per_cm > 0 else 0.0,
                    confidence=round(weighted_confidence, 1),
                    samples=total_samples
                )
            else:
                overall_average = ScaleResult(
                    points_per_cm=0.0,
                    cm_per_point=0.0,
                    points_per_m=0.0,
                    m_per_point=0.0,
                    confidence=0.0,
                    samples=0
                )
        else:
            overall_average = ScaleResult(
                points_per_cm=0.0,
                cm_per_point=0.0,
                points_per_m=0.0,
                m_per_point=0.0,
                confidence=0.0,
                samples=0
            )
        
        # Build response
        response = ScaleOutput(
            drawing_type=input_data.drawing_type,
            regions=region_results,
            overall_average=overall_average,
            total_valid_pairs=all_valid_pairs
        )
        
        logger.info(f"Scale calculation completed: {all_valid_pairs} total valid pairs")
        logger.info(f"Overall scale: {overall_average.points_per_cm:.2f} pt/cm (confidence: {overall_average.confidence:.1f}%)")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during scale calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Region-Based Scale Calculation API",
        "version": "1.0.0",
        "description": "Calculates scale per region from filtered vector data",
        "endpoints": {
            "/": "This page",
            "/calculate-region-scales/": "POST - Calculate scale for each region",
            "/health/": "GET - Health check"
        },
        "workflow": [
            "1. Receives filtered data with regions containing lines and texts",
            "2. For each region, matches dimension texts to nearest lines",
            "3. Calculates scale by dividing line length by dimension value",
            "4. Returns scale per region and overall average"
        ],
        "matching_logic": [
            "Uses line midpoint to find nearest dimension text",
            "Calculates perpendicular distance from text to line",
            "Matches within threshold (10% of average line length)",
            "Filters outliers using z-score method"
        ]
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "port": PORT
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Region-Based Scale API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
