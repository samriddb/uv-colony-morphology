#!/bin/bash

# Create main results directory
mkdir -p results

# Function to extract condition info from filename
get_condition() {
    filename=$(basename "$1")
    if [[ $filename =~ ([0-9]+)p-([0-9]+)s-c([0-9]+)-(pre|post)\.png ]]; then
        plate="${BASH_REMATCH[1]}"
        seconds="${BASH_REMATCH[2]}"
        condition="${BASH_REMATCH[3]}"
        stage="${BASH_REMATCH[4]}"
        echo "${seconds}s_plate${plate}"
    elif [[ $filename =~ ([0-9]+)p-ctrl-c([0-9]+)-(pre|post)\.png ]]; then
        plate="${BASH_REMATCH[1]}"
        echo "ctrl_plate${plate}"
    else
        echo "unknown"
    fi
}

# Function to run analysis and organize output
run_analysis() {
    pre_file="$1"
    post_file="$2"
    
    echo "========================================"
    echo "Processing: $pre_file -> $post_file"
    echo "========================================"
    
    # Extract condition from filename
    condition=$(get_condition "$pre_file")
    
    if [ "$condition" = "unknown" ]; then
        echo "Warning: Could not parse condition from $pre_file"
        condition="unknown_$(date +%s)"
    fi
    
    # Create condition-specific output directory
    output_dir="results/$condition"
    mkdir -p "$output_dir"
    
    # Run the analysis
    python colony_analyzer.py --pre "$pre_file" --post "$post_file"
    
    # Check if analysis succeeded
    if [ -d "outputs" ] && [ "$(ls -A outputs 2>/dev/null)" ]; then
        echo "Moving results to: $output_dir"
        mv outputs/* "$output_dir/"
        echo "✓ Analysis complete for $condition"
    else
        echo "✗ No outputs generated for $condition"
    fi
    
    echo ""
}

# Main processing
echo "Starting batch colony analysis..."
echo "Results will be organized in ./results/"
echo ""

# Process all image pairs
declare -A pairs=(
    ["pre-uv/1p-0s-c12-pre.png"]="post-uv/1p-0s-c12-post.png"
    ["pre-uv/2p-0s-c12-pre.png"]="post-uv/2p-0s-c12-post.png"
    ["pre-uv/3p-10s-c12-pre.png"]="post-uv/3p-10s-c12-post.png"
    ["pre-uv/4p-10s-c12-pre.png"]="post-uv/4p-10s-c12-post.png"
    ["pre-uv/5p-30s-c12-pre.png"]="post-uv/5p-30s-c12-post.png"
    ["pre-uv/6p-30s-c12-pre.png"]="post-uv/6p-30s-c12-post.png"
    ["pre-uv/7p-60s-c12-pre.png"]="post-uv/7p-60s-c12-post.png"
    ["pre-uv/8p-60s-c12-pre.png"]="post-uv/8p-60s-c12-post.png"
    ["pre-uv/9p-ctrl-c12-pre.png"]="post-uv/9p-ctrl-c12-post.png"
)

# Run analysis for each pair
for pre in "${!pairs[@]}"; do
    post="${pairs[$pre]}"
    
    # Check if files exist
    if [ -f "$pre" ] && [ -f "$post" ]; then
        run_analysis "$pre" "$post"
    else
        echo "Warning: Missing files - $pre or $post"
    fi
done

echo "========================================"
echo "Batch processing complete!"
echo "Check ./results/ for organized outputs:"
echo ""
ls -la results/
echo "========================================"