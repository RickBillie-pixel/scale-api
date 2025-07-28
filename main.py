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

# Constants - ENHANCED BASED ON ANALYSIS
PORT = int(os.environ.get("PORT", 10000))

# 1. MINIMUM DIMENSION THRESHOLDS (filter kleine/onbetrouwbare dimensies)
MIN_DIMENSION_THRESHOLDS = {
    "plattegrond": {
        "horizontal": 1000,  # mm - minimaal 1000mm voor horizontale dimensies
        "vertical": 1000     # mm - minimaal 1000mm voor verticale dimensies  
    },
    "bestektekening": {
        "horizontal": 2000,  # mm - minimaal 2000mm (4200, 4600 etc zijn OK)
        "vertical": 800      # mm - minimaal 800mm (930 is OK, maar 420 niet)
    },
    "doorsnede": {
        "horizontal": 1500,  # mm - minimaal 1500mm
        "vertical": 800      # mm - minimaal 800mm (filter 420mm weg)
    },
    "gevelaanzicht": {
        "horizontal": 1000,  # mm 
        "vertical": 800      # mm
    },
    "detailtekening": {
        "horizontal": 500,   # mm - kleinere details toegestaan
        "vertical": 500      # mm
    },
    "installatietekening": {
        "horizontal": 1000,  # mm
        "vertical": 800      # mm
    }
}

# 2. ENHANCED DISTANCE THRESHOLDS
DISTANCE_THRESHOLDS = {
    "normal": 15.0,     # pt - verhoog van 10 naar 15pt
    "fallback": 25.0    # pt - maximum fallback distance
}

# 3. MINIMUM LINE LENGTH THRESHOLDS (filter korte lijnen)
MIN_LINE_LENGTH_THRESHOLDS = {
    "plattegrond": 80,       # pt
    "bestektekening": 80,    # pt
    "doorsnede": 80,         # pt - filter ~68pt lijnen weg
    "gevelaanzicht": 60,     # pt
    "detailtekening": 40,    # pt
    "installatietekening": 50 # pt
}

# 4. SCALE VALIDATION RANGES (aangepast op basis van output)
SCALE_RANGES = {
    "plattegrond": {"min": 0.045, "max": 0.075},      # Rond 0.0567
    "bestektekening": {"min": 0.045, "max": 0.075},   # Rond 0.0567
    "doorsnede": {"min": 0.040, "max": 0.080},        # Meer variatie
    "gevelaanzicht": {"min": 0.045, "max": 0.075},    
    "detailtekening": {"min": 0.030, "max": 0.100},   
    "installatietekening": {"min": 0.040, "max": 0.080}
}

# 5. OUTLIER DETECTION RULES
OUTLIER_RULES = {
    "max_scale_pt_per_mm": 0.12,     # Alles boven 0.12 is verdacht
    "min_scale_pt_per_mm": 0.025,    # Alles onder 0.025 is verdacht
    "max_variation_within_region": 0.03,  # Max std dev binnen 1 region
    "dimension_line_ratio_max": 40,   # Max mm/pt ratio
    "dimension_line_ratio_min": 12    # Min mm/pt ratio
}

# 6. DEFAULT SCALES (voor fallback)
DEFAULT_SCALES = {
    "plattegrond": 0.057,
    "bestektekening": 0.057,
    "doorsnede": 0.060,
    "gevelaanzicht": 0.055,
    "detailtekening": 0.060,
    "installatietekening": 0.050
}

app = FastAPI(
    title="Enhanced Scale API",
    description="Advanced scale calculation with outlier detection and validation",
    version="8.0.0"
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

# Enhanced Output Models
class ValidationInfo(BaseModel):
    status: str  # "valid", "outlier", "filtered_out", "suspect"
    reason: str
    threshold_used: Optional[float] = None

class DimensionData(BaseModel):
    value: float
    unit: str
    value_mm: float
    text: str
    validation: ValidationInfo

class LineMatch(BaseModel):
    line: FilteredLine
    distance: float
    validation: ValidationInfo

class EnhancedScaleMatch(BaseModel):
    dimension: DimensionData
    line_match: LineMatch
    scale_pt_per_mm: float
    scale_mm_per_pt: float
    ratio_mm_per_pt: float
    confidence: float
    validation: ValidationInfo

class RegionScaleResult(BaseModel):
    region_label: str
    horizontal: List[EnhancedScaleMatch] = []
    vertical: List[EnhancedScaleMatch] = []
    
    # Statistics
    total_calculations: int = 0
    filtered_small_dimensions: int = 0
    filtered_short_lines: int = 0
    outliers_removed: int = 0
    
    # Region averages
    horizontal_average: Optional[float] = None
    vertical_average: Optional[float] = None
    scales_consistent: bool = False
    scale_deviation: Optional[float] = None
    
    # Final result
    average_scale_pt_per_mm: Optional[float] = None
    average_scale_mm_per_pt: Optional[float] = None
    confidence: float = 0.0
    validation_status: str = "no_data"
    quality_flags: List[str] = []

class EnhancedScaleOutput(BaseModel):
    drawing_type: str
    regions: List[RegionScaleResult]
    
    # Global statistics
    total_regions_processed: int
    total_calculations: int
    total_filtered_dimensions: int
    total_filtered_lines: int
    total_outliers_removed: int
    
    # Global averages
    overall_average_pt_per_mm: Optional[float] = None
    overall_average_mm_per_pt: Optional[float] = None
    
    # Quality assessment
    high_confidence_regions: int = 0
    medium_confidence_regions: int = 0
    low_confidence_regions: int = 0
    
    # Processing info
    validation_rules_applied: Dict[str, Any] = {}
    timestamp: str

# Utility functions
def extract_dimension_with_validation(text: str, drawing_type: str, orientation: str) -> Optional[DimensionData]:
    """Extract and validate dimension"""
    text_clean = text.strip()
    
    # Enhanced pattern matching including +3420P, +6410P formats
    patterns = [
        r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$',
        r'^\+(\d+(?:[,.]\d+)?)\s*[pP]?\s*(mm|cm|m)?$',  # +3420P format
        r'^(\d+(?:[,.]\d+)?)\s*[pP]\s*(mm|cm|m)?$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text_clean)
        if match:
            try:
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else 'mm'
                
                # Convert to mm
                conversions = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
                value_mm = value * conversions.get(unit, 1.0)
                
                # Validation
                min_threshold = MIN_DIMENSION_THRESHOLDS.get(drawing_type, {}).get(orientation, 500)
                
                if value_mm < min_threshold:
                    validation = ValidationInfo(
                        status="filtered_out",
                        reason=f"below_minimum_threshold",
                        threshold_used=min_threshold
                    )
                    logger.debug(f"Filtered small dimension: {text_clean} = {value_mm}mm < {min_threshold}mm")
                    return DimensionData(
                        value=value, unit=unit, value_mm=value_mm, text=text_clean,
                        validation=validation
                    )
                
                # Check for suspicious patterns
                if "+3420P" in text_clean or "+6410P" in text_clean:
                    validation = ValidationInfo(
                        status="suspect",
                        reason="level_indicator_pattern",
                        threshold_used=None
                    )
                    logger.warning(f"Suspect dimension pattern: {text_clean}")
                else:
                    validation = ValidationInfo(
                        status="valid",
                        reason="passed_validation",
                        threshold_used=min_threshold
                    )
                
                return DimensionData(
                    value=value, unit=unit, value_mm=value_mm, text=text_clean,
                    validation=validation
                )
                
            except (ValueError, IndexError):
                continue
    
    return None

def validate_line_length(line: FilteredLine, drawing_type: str) -> ValidationInfo:
    """Validate line length against thresholds"""
    min_length = MIN_LINE_LENGTH_THRESHOLDS.get(drawing_type, 50)
    
    if line.length < min_length:
        return ValidationInfo(
            status="filtered_out",
            reason="line_too_short",
            threshold_used=min_length
        )
    
    return ValidationInfo(
        status="valid",
        reason="adequate_length",
        threshold_used=min_length
    )

def calculate_distance(p1: Dict[str, float], p2: CleanPoint) -> float:
    """Calculate Euclidean distance"""
    return math.sqrt((p2.x - p1["x"])**2 + (p2.y - p1["y"])**2)

def validate_scale(scale_pt_per_mm: float, drawing_type: str, dimension_mm: float, line_length_pt: float) -> ValidationInfo:
    """Enhanced scale validation"""
    
    # Check basic range
    expected_range = SCALE_RANGES.get(drawing_type, {"min": 0.03, "max": 0.10})
    
    if scale_pt_per_mm < OUTLIER_RULES["min_scale_pt_per_mm"]:
        return ValidationInfo(
            status="outlier",
            reason=f"scale_too_low_global",
            threshold_used=OUTLIER_RULES["min_scale_pt_per_mm"]
        )
    
    if scale_pt_per_mm > OUTLIER_RULES["max_scale_pt_per_mm"]:
        return ValidationInfo(
            status="outlier", 
            reason=f"scale_too_high_global",
            threshold_used=OUTLIER_RULES["max_scale_pt_per_mm"]
        )
    
    # Check dimension/line ratio
    ratio_mm_per_pt = dimension_mm / line_length_pt
    if ratio_mm_per_pt > OUTLIER_RULES["dimension_line_ratio_max"]:
        return ValidationInfo(
            status="outlier",
            reason="dimension_line_ratio_too_high",
            threshold_used=OUTLIER_RULES["dimension_line_ratio_max"]
        )
    
    if ratio_mm_per_pt < OUTLIER_RULES["dimension_line_ratio_min"]:
        return ValidationInfo(
            status="outlier",
            reason="dimension_line_ratio_too_low", 
            threshold_used=OUTLIER_RULES["dimension_line_ratio_min"]
        )
    
    # Check drawing type specific range
    if expected_range["min"] <= scale_pt_per_mm <= expected_range["max"]:
        return ValidationInfo(
            status="valid",
            reason="within_expected_range",
            threshold_used=None
        )
    else:
        return ValidationInfo(
            status="suspect",
            reason=f"outside_expected_range_{expected_range['min']}-{expected_range['max']}",
            threshold_used=None
        )

def calculate_confidence(distance: float, line_length: float, scale_validation: str) -> float:
    """Enhanced confidence calculation"""
    # Distance component (0-1, lower distance = higher score)
    distance_score = max(0, 1 - (distance / DISTANCE_THRESHOLDS["normal"]))
    
    # Line length component (0-1, longer lines = higher score)
    length_score = min(1, line_length / 200)  # 200pt = full score
    
    # Validation component
    validation_multipliers = {
        "valid": 1.0,
        "suspect": 0.7,
        "outlier": 0.3,
        "filtered_out": 0.1
    }
    validation_score = validation_multipliers.get(scale_validation, 0.5)
    
    # Weighted combination
    confidence = (0.4 * distance_score + 0.3 * length_score + 0.3) * validation_score
    
    return round(confidence * 100, 1)

def find_best_matches(region: RegionData, drawing_type: str) -> Tuple[List[Dict], List[Dict]]:
    """Find best dimension-line matches with enhanced filtering"""
    
    horizontal_matches = []
    vertical_matches = []
    
    # Process each text for valid dimensions
    for text in region.texts:
        # Try both orientations
        for orientation in ["horizontal", "vertical"]:
            dimension = extract_dimension_with_validation(text.text, drawing_type, orientation)
            
            if not dimension or dimension.validation.status == "filtered_out":
                continue
            
            # Find matching lines
            matching_lines = [line for line in region.lines if line.orientation == orientation]
            
            for line in matching_lines:
                # Validate line length
                line_validation = validate_line_length(line, drawing_type)
                if line_validation.status == "filtered_out":
                    continue
                
                # Calculate distance
                distance = calculate_distance(text.midpoint, line.midpoint)
                
                # Apply distance threshold with fallback
                is_fallback = False
                if distance <= DISTANCE_THRESHOLDS["normal"]:
                    distance_validation = ValidationInfo(status="valid", reason="within_normal_threshold")
                elif distance <= DISTANCE_THRESHOLDS["fallback"]:
                    distance_validation = ValidationInfo(status="suspect", reason="fallback_threshold_used")
                    is_fallback = True
                else:
                    continue  # Skip if too far
                
                # Calculate scale
                scale_pt_per_mm = line.length / dimension.value_mm
                scale_mm_per_pt = 1 / scale_pt_per_mm
                ratio_mm_per_pt = dimension.value_mm / line.length
                
                # Validate scale
                scale_validation = validate_scale(scale_pt_per_mm, drawing_type, dimension.value_mm, line.length)
                
                # Skip outliers
                if scale_validation.status == "outlier":
                    logger.warning(f"Outlier removed: {dimension.text} -> {scale_pt_per_mm:.4f} pt/mm ({scale_validation.reason})")
                    continue
                
                # Calculate confidence
                confidence = calculate_confidence(distance, line.length, scale_validation.status)
                
                # Create match
                match = {
                    'dimension': dimension,
                    'line': line,
                    'distance': distance,
                    'scale_pt_per_mm': scale_pt_per_mm,
                    'scale_mm_per_pt': scale_mm_per_pt,
                    'ratio_mm_per_pt': ratio_mm_per_pt,
                    'confidence': confidence,
                    'distance_validation': distance_validation,
                    'scale_validation': scale_validation,
                    'is_fallback': is_fallback
                }
                
                if orientation == "horizontal":
                    horizontal_matches.append(match)
                else:
                    vertical_matches.append(match)
    
    # Sort by confidence (best first)
    horizontal_matches.sort(key=lambda x: x['confidence'], reverse=True)
    vertical_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Take top 3 per orientation
    return horizontal_matches[:3], vertical_matches[:3]

def detect_and_remove_outliers(scales: List[float]) -> Tuple[List[float], int]:
    """Remove statistical outliers from scale list"""
    if len(scales) < 3:
        return scales, 0
    
    scales_array = np.array(scales)
    mean_scale = np.mean(scales_array)
    std_scale = np.std(scales_array)
    
    # Remove values more than 2 standard deviations from mean
    threshold = 2 * std_scale
    filtered_scales = []
    outliers_removed = 0
    
    for scale in scales:
        if abs(scale - mean_scale) <= threshold:
            filtered_scales.append(scale)
        else:
            outliers_removed += 1
            logger.warning(f"Statistical outlier removed: {scale:.4f} (mean: {mean_scale:.4f}, std: {std_scale:.4f})")
    
    return filtered_scales, outliers_removed

def process_region_enhanced(region: RegionData, drawing_type: str) -> RegionScaleResult:
    """Process region with enhanced validation and filtering"""
    
    logger.info(f"Processing region: {region.label}")
    
    result = RegionScaleResult(
        region_label=region.label,
        validation_status="processing"
    )
    
    # Find matches with enhanced filtering
    horizontal_matches, vertical_matches = find_best_matches(region, drawing_type)
    
    # Convert to enhanced scale matches
    for match in horizontal_matches:
        line_match = LineMatch(
            line=match['line'],
            distance=round(match['distance'], 2),
            validation=match['distance_validation']
        )
        
        enhanced_match = EnhancedScaleMatch(
            dimension=match['dimension'],
            line_match=line_match,
            scale_pt_per_mm=round(match['scale_pt_per_mm'], 4),
            scale_mm_per_pt=round(match['scale_mm_per_pt'], 4),
            ratio_mm_per_pt=round(match['ratio_mm_per_pt'], 2),
            confidence=match['confidence'],
            validation=match['scale_validation']
        )
        result.horizontal.append(enhanced_match)
    
    for match in vertical_matches:
        line_match = LineMatch(
            line=match['line'],
            distance=round(match['distance'], 2),
            validation=match['distance_validation']
        )
        
        enhanced_match = EnhancedScaleMatch(
            dimension=match['dimension'],
            line_match=line_match,
            scale_pt_per_mm=round(match['scale_pt_per_mm'], 4),
            scale_mm_per_pt=round(match['scale_mm_per_pt'], 4),
            ratio_mm_per_pt=round(match['ratio_mm_per_pt'], 2),
            confidence=match['confidence'],
            validation=match['scale_validation']
        )
        result.vertical.append(enhanced_match)
    
    # Calculate statistics
    result.total_calculations = len(result.horizontal) + len(result.vertical)
    
    # Calculate averages per orientation
    if result.horizontal:
        h_scales = [m.scale_pt_per_mm for m in result.horizontal if m.validation.status == "valid"]
        if h_scales:
            h_scales_clean, h_outliers = detect_and_remove_outliers(h_scales)
            if h_scales_clean:
                result.horizontal_average = round(np.mean(h_scales_clean), 4)
                result.outliers_removed += h_outliers
    
    if result.vertical:
        v_scales = [m.scale_pt_per_mm for m in result.vertical if m.validation.status == "valid"]
        if v_scales:
            v_scales_clean, v_outliers = detect_and_remove_outliers(v_scales)
            if v_scales_clean:
                result.vertical_average = round(np.mean(v_scales_clean), 4)
                result.outliers_removed += v_outliers
    
    # Calculate final average with preference logic
    if result.horizontal_average and result.vertical_average:
        # Check consistency
        deviation = abs(result.vertical_average - result.horizontal_average) / result.horizontal_average
        result.scale_deviation = round(deviation * 100, 1)
        result.scales_consistent = deviation < 0.15  # 15% threshold
        
        if result.scales_consistent:
            # Use combined average
            all_valid_scales = []
            all_valid_scales.extend([m.scale_pt_per_mm for m in result.horizontal if m.validation.status == "valid"])
            all_valid_scales.extend([m.scale_pt_per_mm for m in result.vertical if m.validation.status == "valid"])
            clean_scales, additional_outliers = detect_and_remove_outliers(all_valid_scales)
            result.outliers_removed += additional_outliers
            
            if clean_scales:
                result.average_scale_pt_per_mm = round(np.mean(clean_scales), 4)
                result.average_scale_mm_per_pt = round(1 / result.average_scale_pt_per_mm, 4)
                result.quality_flags.append("consistent_scales")
        else:
            # Prefer horizontal (more reliable)
            result.average_scale_pt_per_mm = result.horizontal_average
            result.average_scale_mm_per_pt = round(1 / result.horizontal_average, 4)
            result.quality_flags.append("horizontal_preferred")
            
    elif result.horizontal_average:
        result.average_scale_pt_per_mm = result.horizontal_average
        result.average_scale_mm_per_pt = round(1 / result.horizontal_average, 4)
        result.quality_flags.append("horizontal_only")
        
    elif result.vertical_average:
        result.average_scale_pt_per_mm = result.vertical_average
        result.average_scale_mm_per_pt = round(1 / result.vertical_average, 4)
        result.quality_flags.append("vertical_only")
    else:
        # Use default
        default_scale = DEFAULT_SCALES.get(drawing_type, 0.057)
        result.average_scale_pt_per_mm = default_scale
        result.average_scale_mm_per_pt = round(1 / default_scale, 4)
        result.quality_flags.append("default_used")
    
    # Set validation status and confidence
    if result.average_scale_pt_per_mm:
        avg_confidence = 0
        if result.horizontal or result.vertical:
            all_confidences = [m.confidence for m in result.horizontal + result.vertical]
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        result.confidence = round(avg_confidence, 1)
        
        if avg_confidence >= 80:
            result.validation_status = "high_confidence"
        elif avg_confidence >= 60:
            result.validation_status = "medium_confidence"
        elif avg_confidence >= 40:
            result.validation_status = "low_confidence"
        else:
            result.validation_status = "poor_confidence"
    else:
        result.validation_status = "no_valid_data"
    
    logger.info(f"  {region.label}: {result.total_calculations} calculations, "
                f"average: {result.average_scale_pt_per_mm} pt/mm, "
                f"confidence: {result.confidence}%")
    
    return result

@app.post("/calculate-scale/", response_model=EnhancedScaleOutput)
async def calculate_scale_enhanced(input_data: FilteredInput, debug: bool = Query(False)):
    """Enhanced scale calculation with validation and outlier detection"""
    try:
        logger.info(f"=== Enhanced Scale Calculation Start ===")
        logger.info(f"Processing {input_data.drawing_type} with {len(input_data.regions)} regions")
        
        region_results = []
        valid_scales = []
        total_stats = {
            'calculations': 0,
            'filtered_dimensions': 0,
            'filtered_lines': 0,
            'outliers_removed': 0
        }
        
        # Process each region
        for region in input_data.regions:
            if not region.lines or not region.texts:
                logger.warning(f"Skipping region {region.label} - no lines or texts")
                continue
            
            region_result = process_region_enhanced(region, input_data.drawing_type)
            region_results.append(region_result)
            
            # Collect statistics
            total_stats['calculations'] += region_result.total_calculations
            total_stats['outliers_removed'] += region_result.outliers_removed
            
            # Collect valid scales for overall average
            if (region_result.average_scale_pt_per_mm and 
                region_result.validation_status in ["high_confidence", "medium_confidence"]):
                valid_scales.append(region_result.average_scale_pt_per_mm)
        
        # Calculate overall average
        overall_avg_pt_per_mm = None
        overall_avg_mm_per_pt = None
        
        if valid_scales:
            # Remove outliers from global average
            clean_global_scales, global_outliers = detect_and_remove_outliers(valid_scales)
            total_stats['outliers_removed'] += global_outliers
            
            if clean_global_scales:
                overall_avg_pt_per_mm = round(np.mean(clean_global_scales), 4)
                overall_avg_mm_per_pt = round(1 / overall_avg_pt_per_mm, 4)
        
        # Count confidence levels
        high_conf = len([r for r in region_results if r.validation_status == "high_confidence"])
        medium_conf = len([r for r in region_results if r.validation_status == "medium_confidence"])
        low_conf = len([r for r in region_results if r.validation_status in ["low_confidence", "poor_confidence"]])
        
        # Build validation rules info
        validation_rules = {
            "min_dimension_thresholds": MIN_DIMENSION_THRESHOLDS,
            "distance_thresholds": DISTANCE_THRESHOLDS,
            "min_line_lengths": MIN_LINE_LENGTH_THRESHOLDS,
            "scale_ranges": SCALE_RANGES,
            "outlier_rules": OUTLIER_RULES,
            "statistical_outlier_detection": "2_standard_deviations",
            "horizontal_preference": "15%_deviation_threshold"
        }
        
        # Build response
        response = EnhancedScaleOutput(
            drawing_type=input_data.drawing_type,
            regions=region_results,
            total_regions_processed=len(region_results),
            total_calculations=total_stats['calculations'],
            total_filtered_dimensions=total_stats['filtered_dimensions'],
            total_filtered_lines=total_stats['filtered_lines'], 
            total_outliers_removed=total_stats['outliers_removed'],
            overall_average_pt_per_mm=overall_avg_pt_per_mm,
            overall_average_mm_per_pt=overall_avg_mm_per_pt,
            high_confidence_regions=high_conf,
            medium_confidence_regions=medium_conf,
            low_confidence_regions=low_conf,
            validation_rules_applied=validation_rules,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"\n=== Enhanced Scale Calculation Complete ===")
        logger.info(f"Overall average: {overall_avg_pt_per_mm} pt/mm")
        logger.info(f"High confidence regions: {high_conf}")
        logger.info(f"Total outliers removed: {total_stats['outliers_removed']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during enhanced scale calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with enhanced API information"""
    return {
        "title": "Enhanced Scale API",
        "version": "8.0.0",
        "description": "Advanced scale calculation with outlier detection and validation",
        "enhancements": {
            "dimension_filtering": {
                "plattegrond": "Min 1000mm horizontal/vertical",
                "bestektekening": "Min 2000mm horizontal, 800mm vertical", 
                "doorsnede": "Min 1500mm horizontal, 800mm vertical (filters 420mm)",
                "purpose": "Remove small dimensions that cause extreme scales"
            },
            "line_filtering": {
                "minimum_lengths": MIN_LINE_LENGTH_THRESHOLDS,
                "purpose": "Filter short lines (~68pt) that cause unreliable scales"
            },
            "outlier_detection": {
                "global_limits": "0.025 - 0.12 pt/mm",
                "ratio_limits": "12 - 40 mm/pt",
                "statistical": "2 standard deviations",
                "purpose": "Remove extreme values like 0.1821 and 0.0106 pt/mm"
            },
            "enhanced_validation": {
                "confidence_scoring": "Distance + length + validation quality",
                "scale_consistency": "15% deviation threshold",
                "horizontal_preference": "When vertical scales inconsistent",
                "quality_flags
