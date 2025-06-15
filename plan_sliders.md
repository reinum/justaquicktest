# Slider Implementation Plan for osu! AI Replay Maker

## Problem Statement

The current AI model cannot properly handle sliders because:
1. **No slider path knowledge**: Only has control points, not the actual curved path
2. **No slider ball position**: Doesn't know where the slider ball should be at any given time
3. **Missing temporal information**: No understanding of slider progression over time

## Solution Overview

Implement a comprehensive slider path calculation system that:
1. Calculates smooth slider paths from control points using proper curve algorithms
2. Provides slider ball position as a percentage of completion
3. Integrates slider information into the AI model's training data

## Technical Research Findings

### osu! Slider Curve Types

Based on research, osu! supports four main curve types:

1. **Bezier Curves** (Most common)
   - Uses De Casteljau's algorithm
   - Handles arbitrary number of control points
   - Supports curve segments with repeated points

2. **Perfect Circle Curves**
   - For 3-point sliders forming circular arcs
   - Calculates circle passing through all three points
   - Extends as circular arc if length exceeds natural curve

3. **Catmull-Rom Splines**
   - Smooth interpolation through control points
   - Less common, mostly legacy
   - Creates flowing curves

4. **Linear Curves**
   - Straight line segments between points
   - Simplest case

### Mathematical Foundation

**De Casteljau's Algorithm for Bezier Curves:**
```
function bezier_point(control_points, t):
    if len(control_points) == 1:
        return control_points[0]
    
    new_points = []
    for i in range(len(control_points) - 1):
        point = (1 - t) * control_points[i] + t * control_points[i + 1]
        new_points.append(point)
    
    return bezier_point(new_points, t)
```

**Enhanced Slider Ball Tracking:**
```
# Position tracking
progress_percentage = (current_time - slider_start_time) / slider_duration
slider_ball_position = calculate_curve_point(progress_percentage)

# Velocity tracking  
time_remaining = slider_end_time - current_time
path_length_remaining = calculate_remaining_path_length(progress_percentage)
target_velocity = path_length_remaining / max(time_remaining, 1)  # pixels/ms

# Acceleration for smooth movement
velocity_change = target_velocity - previous_velocity
target_acceleration = velocity_change / frame_time_delta
```

## Implementation Plan

### Phase 1: Core Slider Path System

#### 1.1 Create `slider_path.py` Module

**Base Classes:**
```python
class SliderCurve:
    def calculate_point(self, t: float) -> tuple[float, float]
    def calculate_length(self) -> float
    def get_path_points(self, num_points: int) -> list[tuple[float, float]]

class BezierCurve(SliderCurve):
    # De Casteljau's algorithm implementation
    
class PerfectCircle(SliderCurve):
    # Circle calculation through 3 points
    
class CatmullRom(SliderCurve):
    # Catmull-Rom spline implementation
    
class LinearCurve(SliderCurve):
    # Linear interpolation
```

**Main Calculator:**
```python
class SliderPathCalculator:
    def __init__(self, hit_object_data):
        self.curve_type = self.determine_curve_type(hit_object_data)
        self.control_points = hit_object_data['control_points']
        self.curve = self.create_curve()
    
    def get_slider_ball_position(self, progress_percentage: float) -> tuple[float, float]
    def get_path_visualization(self, num_points: int = 100) -> list[tuple[float, float]]
```

#### 1.2 Curve Type Detection

Implement logic to determine curve type from osu! file format:
- Parse curve type indicators (B, P, C, L)
- Handle mixed curve segments
- Default fallback to Bezier

#### 1.3 Path Length Calculation

Accurate path length calculation for timing:
- Numerical integration for complex curves
- Analytical solutions where possible
- Cache results for performance

### Phase 2: Model Integration

#### 2.1 Data Preprocessing Enhancement

**Modify data loader to include:**
```python
class SliderAwareDataLoader:
    def preprocess_slider_data(self, hit_objects):
        for hit_object in hit_objects:
            if hit_object.type == 'slider':
                # Calculate slider path
                path_calculator = SliderPathCalculator(hit_object)
                
                # Generate path points
                path_points = path_calculator.get_path_visualization(50)
                
                # Store slider metadata
                hit_object.slider_path = path_points
                hit_object.slider_duration = hit_object.end_time - hit_object.start_time
```

#### 2.2 Enhanced Feature Engineering

**Comprehensive slider feature set:**

**Position Features:**
1. **Slider Progress**: 0-1 value indicating completion percentage
2. **Target Position**: (x, y) coordinates where slider ball should be
3. **Path Points**: Pre-calculated path points as spatial features

**Velocity Features:**
4. **Target Velocity**: Required velocity in pixels/ms to complete slider on time
5. **Current Velocity**: Actual cursor velocity from previous frame
6. **Velocity Error**: Difference between target and current velocity

**Temporal Features:**
7. **Time Remaining**: Milliseconds left to complete slider
8. **Time Elapsed**: Milliseconds since slider started
9. **Urgency Factor**: time_remaining / total_slider_duration

**Geometric Features:**
10. **Curve Complexity**: Measure of upcoming path curvature
11. **Direction Change**: Upcoming direction changes in degrees
12. **Path Segment Type**: Current curve type (bezier, linear, etc.)

**Context Features:**
13. **Slider Active**: Boolean indicating if slider is currently active
14. **BPM**: Current beats per minute
15. **Slider Velocity**: Current SV multiplier
16. **Slider Multiplier**: Base slider speed multiplier

#### 2.3 Model Architecture Updates

**Expanded Input Feature Set:**
```python
# Current features + comprehensive slider features
input_features = {
    # Existing features
    'cursor_position': (x, y),
    'hit_objects': [...],
    
    # Position features
    'slider_progress': 0.0-1.0,
    'target_slider_pos': (x, y),
    'slider_path_points': [...],
    
    # Velocity features  
    'target_velocity': float,     # pixels/ms
    'current_velocity': float,    # pixels/ms
    'velocity_error': float,      # target - current
    
    # Temporal features
    'time_remaining': float,      # ms
    'time_elapsed': float,        # ms  
    'urgency_factor': 0.0-1.0,    # time pressure
    
    # Geometric features
    'curve_complexity': float,    # curvature measure
    'direction_change': float,    # upcoming angle change
    'path_segment_type': int,     # curve type enum
    
    # Context features
    'slider_active': bool,
    'current_bpm': float,
    'slider_velocity': float,     # SV multiplier
    'slider_multiplier': float    # base speed
}
```

**Temporal Modeling:**
- Add LSTM layers to handle slider progression over time
- Attention mechanisms for slider path awareness
- Multi-head attention for different slider types

### Phase 3: Training Data Generation

#### 3.1 Slider-Aware Replay Processing

**Enhanced replay analysis:**
```python
def process_slider_frames(replay_frames, slider_data):
    for frame in replay_frames:
        if frame.time in slider_active_period:
            # Calculate expected slider ball position
            progress = (frame.time - slider.start_time) / slider.duration
            expected_pos = slider_calculator.get_slider_ball_position(progress)
            
            # Add slider features to frame data
            frame.slider_progress = progress
            frame.slider_target = expected_pos
            frame.slider_active = True
```

#### 3.2 Training Objective Updates

**Enhanced Multi-objective Loss Function:**
```python
total_loss = (
    cursor_position_loss +           # Basic position accuracy
    slider_path_following_loss +     # How well cursor follows slider path
    velocity_matching_loss +         # How well velocity matches target
    acceleration_smoothness_loss +   # Penalize jerky movements
    timing_accuracy_loss +           # Click timing precision
    slider_completion_loss +         # Penalty for not completing sliders
    human_likeness_loss             # Deviation from human movement patterns
)
```

### Phase 4: Testing and Validation

#### 4.1 Unit Tests for Curve Calculations

**Test cases:**
- Linear slider: Verify straight line interpolation
- Perfect circle: Test 3-point circle calculation
- Bezier curves: Compare against reference implementations
- Edge cases: Single point, overlapping points, extreme curves

#### 4.2 Integration Tests

**Validation against real osu! maps:**
- Load actual .osu files
- Calculate slider paths
- Compare with expected behavior
- Visual verification tools

#### 4.3 Performance Testing

**Optimization targets:**
- Path calculation speed
- Memory usage for path storage
- Real-time slider ball position updates

### Phase 5: Advanced Features

#### 5.1 Slider Velocity Handling

**Variable slider speeds:**
- Parse slider velocity multipliers
- Adjust timing calculations
- Handle speed changes mid-slider

#### 5.2 Repeat Sliders

**Multi-repeat slider support:**
- Calculate reverse paths
- Handle repeat timing
- Smooth direction transitions

#### 5.3 Visual Debugging Tools

**Development aids:**
- Slider path visualization
- Real-time slider ball tracking
- Comparison with osu! reference

## Implementation Timeline

**Week 1-2: Core Path System**
- Implement basic curve classes
- De Casteljau's algorithm for Bezier
- Linear and perfect circle curves
- Basic testing

**Week 3-4: Model Integration**
- Modify data preprocessing
- Add slider features to model input
- Update training pipeline
- Initial training experiments

**Week 5-6: Testing and Refinement**
- Comprehensive testing suite
- Performance optimization
- Accuracy validation
- Bug fixes and improvements

**Week 7-8: Advanced Features**
- Catmull-Rom splines
- Slider velocity handling
- Repeat sliders
- Visual debugging tools

## Expected Outcomes

1. **Accurate Slider Paths**: AI will understand and follow slider curves properly
2. **Realistic Velocity Control**: AI will move at appropriate speeds for different slider types
3. **Human-like Movement**: Natural acceleration/deceleration patterns
4. **Improved Replay Quality**: More realistic and believable slider gameplay
5. **Better Training Data**: Richer feature representation capturing full slider dynamics
6. **Adaptive Behavior**: AI adapts movement speed to slider complexity and timing
7. **Extensible System**: Easy to add new curve types or movement features

## Technical Challenges and Solutions

**Challenge 1: Curve Accuracy**
- Solution: Use well-tested mathematical algorithms (De Casteljau)
- Validation against osu! reference implementation

**Challenge 2: Performance**
- Solution: Pre-calculate paths, cache results, optimize algorithms
- Use numpy for vectorized operations

**Challenge 3: Feature Complexity**
- Solution: Gradual feature introduction, ablation studies
- Start with position + velocity, add geometric features incrementally
- Use feature importance analysis to identify most valuable features

**Challenge 4: Velocity Calculation Accuracy**
- Solution: Implement precise timing calculations using osu! formula
- Account for BPM changes, SV multipliers, and slider multipliers
- Validate against frame-by-frame analysis of real replays

**Challenge 5: Training Data Quality**
- Solution: Careful validation of slider timing calculations
- Cross-reference with multiple replay sources
- Filter out replays with obvious slider following errors

**Challenge 6: Model Convergence**
- Solution: Use curriculum learning - start with simple sliders
- Gradually introduce complex curves and variable velocities
- Multi-stage training with different loss function weights

## Success Metrics

1. **Path Accuracy**: < 1 pixel deviation from reference implementation
2. **Velocity Accuracy**: < 5% deviation from calculated target velocity
3. **Performance**: < 1ms per slider path calculation
4. **Smoothness**: Low acceleration variance (< 50 pixels/ms²)
5. **Model Improvement**: Measurable increase in slider following accuracy
6. **Timing Precision**: Slider completion within ±10ms of expected time
7. **Human Similarity**: Movement patterns statistically similar to human replays
8. **Replay Quality**: Visual inspection shows realistic and natural slider movement

This plan provides a comprehensive roadmap for implementing proper slider support in the osu! AI replay maker, transforming it from a point-to-point aiming system to a sophisticated curve-following AI that understands the full complexity of osu! gameplay.