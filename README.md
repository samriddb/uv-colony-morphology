# UV Colony Morphology Analysis

Computational analysis pipeline for quantifying UV radiation effects on bacterial colony spatial organization and morphological characteristics.

## Overview

This project analyzes how UV stress alters bacterial colony growth patterns, spatial distribution, and morphological features using automated image processing and network analysis. The pipeline processes pre/post UV exposure images to quantify changes in colony survival, size, clustering, and spatial organization.

**Key Research Question:** Does UV radiation promote colony isolation or clustering patterns in bacterial populations?

## Features

### Image Processing Pipeline
- **Automated plate detection** using Hough circle detection
- **CLAHE enhancement** for improved colony contrast
- **Watershed segmentation** with morphological cleaning
- **Edge artifact removal** and debris filtering

### Quantitative Analysis
- **Morphological features**: Area, diameter, eccentricity, solidity, orientation
- **Spatial analysis**: k-nearest neighbor distances, clustering coefficients
- **Network analysis**: Graph connectivity, community detection, centrality measures
- **Statistical comparison**: Pre vs. post UV treatment metrics

### Visualization Outputs
- Colony segmentation overlays with ID labeling
- Size distribution histograms and statistical summaries
- Spatial network graphs showing colony connectivity
- Nearest neighbor distance analysis plots
- Growth orientation rose plots

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python-headless
- scikit-image
- scikit-learn
- scipy
- networkx
- matplotlib
- numpy
- pandas
- python-louvain

## Usage

### Basic Analysis

**Single plate analysis:**
```bash
python colony_analyzer.py --single path/to/plate_image.png
```

**Pre/Post UV comparison:**
```bash
python colony_analyzer.py --pre path/to/pre_uv.png --post path/to/post_uv.png
```

**Custom k-NN parameter:**
```bash
python colony_analyzer.py --pre pre_image.png --post post_image.png --k 3
```

### Batch Processing

For multiple conditions, use the provided batch script:
```bash
./batch_analyze.sh
```

This processes all UV dose conditions and organizes results by treatment group.

## Methodology

### Image Preprocessing
1. **Plate Detection**: Hough circle detection identifies petri dish boundaries
2. **Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) improves colony visibility
3. **Noise Reduction**: Gaussian blur smooths texture artifacts
4. **ROI Definition**: Analysis restricted to inner 88% of plate radius

### Colony Segmentation
1. **Thresholding**: Otsu's method for automatic binary conversion
2. **Morphological Operations**: Opening/closing to separate touching colonies
3. **Watershed Algorithm**: Distance transform + local maxima for boundary delineation
4. **Post-filtering**: Size thresholds remove debris artifacts

### Feature Extraction
- **Morphological**: Area, perimeter, equivalent diameter, eccentricity, solidity
- **Spatial**: Centroid coordinates, distance to plate center
- **Shape**: Ellipse fitting, convex hull ratio, compactness index
- **Intensity**: Mean pixel brightness within colony boundaries

### Spatial Network Analysis
- **k-Nearest Neighbors**: Identifies colony spatial relationships
- **Graph Construction**: Colonies as nodes, proximity as weighted edges
- **Network Metrics**: Degree distribution, clustering coefficient, betweenness centrality
- **Community Detection**: Louvain algorithm identifies colony clusters

## File Structure

```
uv-colony-morphology/
├── colony_analyzer.py          # Main analysis pipeline
├── requirements.txt            # Python dependencies
├── batch_analyze.sh           # Batch processing script
├── pre-uv/                    # Pre-UV treatment images
├── post-uv/                   # Post-UV treatment images
└── results/                   # Organized analysis outputs
    ├── 0s_plate1/             # Control condition results
    ├── 10s_plate3/            # 10s UV treatment results
    ├── 30s_plate5/            # 30s UV treatment results
    └── 60s_plate7/            # 60s UV treatment results
```

## Output Files

Each analysis generates:
- `*_seg_clean.png` - Clean segmentation overlay
- `*_seg_labeled.png` - Segmentation with colony IDs
- `*_size_dist.png` - Colony size distribution histogram
- `*_knn.png` - Nearest neighbor distance analysis
- `*_graph.png` - Spatial network visualization
- `*_orientation.png` - Colony growth direction analysis
- `comparison.png` - Pre vs. post statistical comparison

## Example Results

### Segmentation Output
Automated colony detection and boundary delineation with unique ID assignment.

### Spatial Analysis
Quantitative measurement of colony clustering vs. isolation patterns using network theory.

### Morphological Changes
Statistical analysis of size distribution shifts and shape changes under UV stress.

## Model Organism

**Bacillus cereus** - Gram-positive, spore-forming bacterium chosen for:
- Clear colony morphology facilitating computational segmentation
- Well-characterized stress response patterns
- Distinctive growth characteristics suitable for spatial analysis

*Note: Originally planned for Serratia marcescens (red pigmentation advantage) but cultures were unavailable.*

## Potential Limitations

- **Segmentation sensitivity**: Parameter tuning required for different imaging conditions
- **Scale dependence**: Pixel-based measurements require consistent camera setup
- **No individual tracking**: Population-level analysis cannot follow specific colonies
- **Processing artifacts**: Over/under-segmentation in challenging conditions

---

*Developed for quantitative analysis of bacterial colony spatial organization under UV stress.*