"""
Optimized Scale Calculation API for Technical CAD Drawings
Calculates precise scale from vector data and dimension texts
"""

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
DISTANCE_THRESHOLD_FACTOR = 0.05  # 5% of average line length
MIN_DISTANCE_THRESHOLD = 50.0
MIN_CONFIDENCE = 0.0
OUTLIER_Z_SCORE = 2.0

app = FastAPI(
    title="Scale Calculation API",
    description="Calculates precise scale from CAD vector data and dimension texts",
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

# Pydantic models
class Point(BaseModel):
    x: float
    y: float

class Vector(BaseModel):
    type: str = "line"
    p1: Point
    p2: Point
    length: Optional[float] = None
    orientation: Optional[str] = None

class Text(BaseModel):
    text: str
    position: Point
    bbox: Optional[Dict[str, float]] = None
    source: Optional[str] = None

class InputData(BaseModel):
    vector_data: List[Vector] = Field(..., alias="vector_data")
    texts: List[Text]

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

class ScaleResult(BaseModel):
    points_per_cm: float
    cm_per_point: float
    points_per_m: float
    m_per_point: float
    confidence: float
    samples: int

class OutputData(BaseModel):
    scale: Dict[str, ScaleResult]
    average: ScaleResult
    unit: str = "cm"
    valid_pairs_used: int
    total_matches_checked: int
    matched_dimensions: List[DimensionMatch]

# Utility functions
def calculate_line_length(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def get_midpoint(p1: Point, p2: Point) -> Point:
    """Calculate midpoint of a line segment"""
    return Point(x=(p1.x + p2.x) / 2, y=(p1.y + p2.y) / 2)

def determine_orientation(p1: Point, p2: Point, angle_threshold: float = 10.0) -> str:
    """
    Determine line orientation (horizontal, vertical, or diagonal)
    angle_threshold: degrees from horizontal/vertical to still consider it aligned
    """
    dx = abs(p2.x - p1.x)
    dy = abs(p2.y - p1.y)
    
    if dx == 0 and dy == 0:
        return "point"
    
    angle = math.degrees(math.atan2(dy, dx))
    
    # Check if near horizontal (0 or 180 degrees)
    if angle < angle_threshold or angle > (180 - angle_threshold):
        return "horizontal"
    # Check if near vertical (90 degrees)
    elif abs(angle - 90) < angle_threshold:
        return "vertical"
    else:
        return "diagonal"

def calculate_perpendicular_distance(line_p1: Point, line_p2: Point, point: Point) -> float:
    """Calculate perpendicular distance from point to line segment"""
    # Vector from p1 to p2
    dx = line_p2.x - line_p1.x
    dy = line_p2.y - line_p1.y
    
    # Vector from p1 to point
    px = point.x - line_p1.x
    py = point.y - line_p1.y
    
    # Line length squared
    line_length_sq = dx * dx + dy * dy
    
    if line_length_sq == 0:
        # Line is a point
        return math.sqrt(px * px + py * py)
    
    # Parameter t of the projection
    t = max(0, min(1, (px * dx + py * dy) / line_length_sq))
    
    # Projection point on the line
    proj_x = line_p1.x + t * dx
    proj_y = line_p1.y + t * dy
    
    # Distance from point to projection
    return math.sqrt((point.x - proj_x)**2 + (point.y - proj_y)**2)

def extract_dimension(text: str) -> Optional[DimensionData]:
    """Extract dimension value and unit from text"""
    # Clean text
    text = text.strip()
    
    # Comprehensive patterns for dimension extraction
    patterns = [
        # With units
        (r'(\d+(?:[.,]\d+)?)\s*(mm|cm|m|meter|metre)\b', 'metric'),
        (r'(\d+(?:[.,]\d+)?)\s*(?:"|″|inch|inches|in)\b', 'inch'),
        (r'(\d+(?:[.,]\d+)?)\s*(?:\'|′|ft|foot|feet)\b', 'feet'),
        
        # Diameter and radius
        (r'[ØΦ⌀]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', 'diameter'),
        (r'R\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?', 'radius'),
        
        # Just numbers (assume mm)
        (r'^(\d+(?:[.,]\d+)?)$', 'number'),
        
        # Numbers with multiplication
        (r'(\d+)\s*[xX×]\s*(\d+(?:[.,]\d+)?)', 'multiply'),
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Extract value (handle comma as decimal separator)
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                # Extract unit or default to mm
                if pattern_type == 'metric' and match.group(2):
                    unit = match.group(2).lower()
                elif pattern_type == 'inch':
                    unit = 'in'
                    value = value  # Keep original value
                elif pattern_type == 'feet':
                    unit = 'ft'
                    value = value  # Keep original value
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
                
                return DimensionData(
                    value=value,
                    unit=unit,
                    value_mm=value_mm
                )
            except (ValueError, IndexError):
                continue
    
    return None

def sigmoid(x: float, steepness: float = 1.0) -> float:
    """Sigmoid function for confidence calculation"""
    return 1 / (1 + math.exp(-steepness * x))

def calculate_confidence(
    samples: int,
    consistency: float,
    avg_distance: float,
    threshold: float
) -> float:
    """Calculate confidence score based on multiple factors"""
    # Sample score (40% weight) - more samples = higher confidence
    sample_score = 40 * sigmoid((samples - 1) / 5.0, steepness=2.0)
    
    # Consistency score (40% weight) - lower variation = higher confidence
    consistency_score = 40 * (1 - consistency) if consistency < 1 else 0
    
    # Distance quality score (20% weight) - closer matches = higher confidence
    distance_ratio = min(1.0, avg_distance / threshold)
    quality_score = 20 * (1 - distance_ratio)
    
    # Total confidence
    confidence = sample_score + consistency_score + quality_score
    
    return max(0, min(100, confidence))

def match_dimensions_to_lines(
    vectors: List[Vector],
    texts: List[Text],
    orientation: str
) -> Tuple[List[Dict], int]:
    """Match dimension texts to their corresponding lines for a specific orientation"""
    matches = []
    total_checked = 0
    
    # Filter vectors by orientation
    oriented_vectors = []
    for vec in vectors:
        # Calculate length if not provided
        if vec.length is None:
            vec.length = calculate_line_length(vec.p1, vec.p2)
        
        # Determine orientation if not provided
        if vec.orientation is None:
            vec.orientation = determine_orientation(vec.p1, vec.p2)
        
        if vec.orientation == orientation and vec.length > 0:
            oriented_vectors.append(vec)
    
    if not oriented_vectors:
        return [], 0
    
    # Calculate dynamic threshold
    avg_line_length = np.mean([v.length for v in oriented_vectors])
    threshold = max(MIN_DISTANCE_THRESHOLD, avg_line_length * DISTANCE_THRESHOLD_FACTOR)
    
    # Match each line with the closest valid dimension text
    for vec in oriented_vectors:
        midpoint = get_midpoint(vec.p1, vec.p2)
        best_match = None
        best_distance = float('inf')
        
        for txt in texts:
            total_checked += 1
            
            # Calculate distance from text to line
            perp_dist = calculate_perpendicular_distance(vec.p1, vec.p2, txt.position)
            
            # Also consider distance to midpoint
            mid_dist = calculate_line_length(midpoint, txt.position)
            
            # Use minimum distance
            min_dist = min(perp_dist, mid_dist)
            
            if min_dist < threshold and min_dist < best_distance:
                # Try to extract dimension
                dim = extract_dimension(txt.text)
                if dim and dim.value_mm > 0:
                    best_distance = min_dist
                    best_match = (txt, dim)
        
        if best_match:
            txt, dim = best_match
            points_per_mm = vec.length / dim.value_mm
            
            matches.append({
                "line": vec,
                "text": txt,
                "dimension": dim,
                "match_distance": best_distance,
                "points_per_mm": points_per_mm
            })
    
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
    
    # Calculate z-scores
    z_scores = [(s - mean_scale) / std_scale for s in scales]
    
    # Filter based on z-score threshold
    filtered_matches = [
        m for i, m in enumerate(matches)
        if abs(z_scores[i]) <= OUTLIER_Z_SCORE
    ]
    
    # Ensure we keep at least 30% of matches
    if len(filtered_matches) < max(1, len(matches) * 0.3):
        # Sort by z-score and keep the best ones
        sorted_indices = sorted(range(len(matches)), key=lambda i: abs(z_scores[i]))
        keep_count = max(1, int(len(matches) * 0.3))
        filtered_matches = [matches[i] for i in sorted_indices[:keep_count]]
    
    return filtered_matches

def calculate_scale_stats(matches: List[Dict], threshold: float) -> ScaleResult:
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
    
    # Extract scales
    scales_mm = [m['points_per_mm'] for m in matches]
    mean_scale_mm = np.mean(scales_mm)
    
    # Calculate consistency (coefficient of variation)
    if len(scales_mm) > 1:
        std_scale_mm = np.std(scales_mm)
        consistency = std_scale_mm / mean_scale_mm if mean_scale_mm > 0 else 1.0
    else:
        consistency = 0.0  # Perfect consistency for single sample
    
    # Calculate average match distance
    avg_distance = np.mean([m['match_distance'] for m in matches])
    
    # Calculate confidence
    confidence = calculate_confidence(
        samples=len(matches),
        consistency=consistency,
        avg_distance=avg_distance,
        threshold=threshold
    )
    
    # Convert to different units
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

@app.post("/extract-scale/", response_model=OutputData)
async def extract_scale(data: InputData):
    """Extract scale from vector data and dimension texts"""
    try:
        # Validate input
        if not data.vector_data or not data.texts:
            raise HTTPException(
                status_code=400,
                detail="Input must include both vector_data and texts"
            )
        
        logger.info(f"Processing {len(data.vector_data)} vectors and {len(data.texts)} texts")
        
        # Process horizontal lines
        horiz_matches, horiz_checked = match_dimensions_to_lines(
            data.vector_data, data.texts, "horizontal"
        )
        horiz_filtered = filter_outliers(horiz_matches)
        
        # Process vertical lines
        vert_matches, vert_checked = match_dimensions_to_lines(
            data.vector_data, data.texts, "vertical"
        )
        vert_filtered = filter_outliers(vert_matches)
        
        # Calculate scale statistics
        avg_line_length = np.mean([
            v.length if v.length else calculate_line_length(v.p1, v.p2)
            for v in data.vector_data
        ])
        threshold = max(MIN_DISTANCE_THRESHOLD, avg_line_length * DISTANCE_THRESHOLD_FACTOR)
        
        horizontal_scale = calculate_scale_stats(horiz_filtered, threshold)
        vertical_scale = calculate_scale_stats(vert_filtered, threshold)
        
        # Calculate average scale
        all_filtered = horiz_filtered + vert_filtered
        if all_filtered:
            # Weighted average based on number of samples
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
                average_scale = calculate_scale_stats(all_filtered, threshold)
        else:
            raise HTTPException(
                status_code=400,
                detail="No valid dimension-line matches found. Ensure dimension texts are near their corresponding lines."
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
                match_distance=round(m['match_distance'], 2)
            ))
        
        # Build response
        scale_dict = {}
        if horizontal_scale.samples > 0:
            scale_dict["horizontal"] = horizontal_scale
        if vertical_scale.samples > 0:
            scale_dict["vertical"] = vertical_scale
        
        response = OutputData(
            scale=scale_dict,
            average=average_scale,
            unit="cm",
            valid_pairs_used=len(all_filtered),
            total_matches_checked=horiz_checked + vert_checked,
            matched_dimensions=matched_dimensions
        )
        
        logger.info(
            f"Scale calculation completed: {response.valid_pairs_used} valid pairs "
            f"from {response.total_matches_checked} checks"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during scale calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Scale Calculation API",
        "version": "2.0.0",
        "description": "Calculates precise scale from CAD vector data and dimension texts",
        "endpoints": {
            "/": "This page",
            "/extract-scale/": "POST - Calculate scale from vector data and texts",
            "/health/": "GET - Health check"
        }
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
    logger.info(f"Starting Scale API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
