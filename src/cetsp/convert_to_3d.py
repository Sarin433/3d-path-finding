"""
CETSP 2D to 3D Converter
========================

Converts standard 2D CETSP datasets to 3D CETSP format by adding
a z-coordinate based on various strategies.

Input format (2D CETSP):
    x y demand radius capacity
    (last two columns sometimes omitted)

Output format (3D CETSP):
    x y z radius
"""

import os
import numpy as np
from pathlib import Path
from typing import Literal, Optional


def parse_2d_cetsp(filepath: str) -> tuple[list[tuple[float, float, float]], dict]:
    """
    Parse a 2D CETSP file.
    
    Returns:
        nodes: List of (x, y, radius) tuples
        metadata: Dictionary with depot info and comments
    """
    nodes = []
    metadata = {'comments': [], 'depot': None}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Capture comments
            if line.startswith('//'):
                metadata['comments'].append(line)
                # Try to extract depot info
                if 'depot' in line.lower():
                    metadata['depot_comment'] = line
                continue
            
            # Parse data line
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    # Third column might be demand (0) or z (if already 3D)
                    # Fourth column is usually radius
                    if len(parts) >= 4:
                        radius = float(parts[3])
                    elif len(parts) >= 3:
                        radius = float(parts[2])
                    else:
                        radius = 10.0  # default
                    
                    nodes.append((x, y, radius))
                except ValueError:
                    continue
    
    return nodes, metadata


def add_z_coordinate(
    nodes: list[tuple[float, float, float]],
    strategy: Literal['wave', 'random', 'dome', 'layers', 'distance'] = 'wave',
    z_min: float = 10.0,
    z_max: float = 90.0,
    seed: Optional[int] = None
) -> list[tuple[float, float, float, float]]:
    """
    Add z-coordinates to 2D nodes using various strategies.
    
    Strategies:
        - 'wave': Sinusoidal wave based on position
        - 'random': Random z values
        - 'dome': Dome shape (higher in center)
        - 'layers': Alternating layer heights
        - 'distance': Based on distance from center
    
    Returns:
        List of (x, y, z, radius) tuples
    """
    rng = np.random.default_rng(seed)
    
    # Get bounds
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    max_dist = np.sqrt((x_max - x_center)**2 + (y_max - y_center)**2)
    
    z_range = z_max - z_min
    z_mid = (z_min + z_max) / 2
    
    result = []
    
    for i, (x, y, radius) in enumerate(nodes):
        if strategy == 'wave':
            # Sinusoidal wave based on position
            norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            z = z_mid + (z_range / 2) * np.sin(norm_x * 2 * np.pi) * np.cos(norm_y * 2 * np.pi)
            
        elif strategy == 'random':
            # Random z values
            z = rng.uniform(z_min, z_max)
            
        elif strategy == 'dome':
            # Dome shape - higher in center
            dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
            norm_dist = dist / max_dist if max_dist > 0 else 0
            z = z_max - (z_range * norm_dist)
            
        elif strategy == 'layers':
            # Alternating layers based on index
            layer = i % 3
            z = z_min + (z_range / 2) * layer
            
        elif strategy == 'distance':
            # Based on distance from center (inverse dome)
            dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
            norm_dist = dist / max_dist if max_dist > 0 else 0
            z = z_min + (z_range * norm_dist)
            
        else:
            z = z_mid
        
        # Clamp z to valid range
        z = max(z_min, min(z_max, z))
        
        result.append((x, y, z, radius))
    
    return result


def convert_cetsp_to_3d(
    input_path: str,
    output_path: Optional[str] = None,
    strategy: Literal['wave', 'random', 'dome', 'layers', 'distance'] = 'wave',
    z_min: float = 10.0,
    z_max: float = 90.0,
    depot_z: float = 50.0,
    seed: Optional[int] = None
) -> str:
    """
    Convert a 2D CETSP file to 3D format.
    
    Args:
        input_path: Path to input 2D CETSP file
        output_path: Path for output 3D file (auto-generated if None)
        strategy: Z-coordinate generation strategy
        z_min: Minimum z value
        z_max: Maximum z value
        depot_z: Z coordinate for depot
        seed: Random seed for reproducibility
    
    Returns:
        Path to the output file
    """
    # Parse input
    nodes, metadata = parse_2d_cetsp(input_path)
    
    if not nodes:
        raise ValueError(f"No valid nodes found in {input_path}")
    
    # Add z coordinates
    nodes_3d = add_z_coordinate(nodes, strategy, z_min, z_max, seed)
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_3d.cetsp")
    
    # Extract depot from original comments or estimate
    depot_x, depot_y = 100.0, 100.0  # default
    for comment in metadata.get('comments', []):
        if 'depot' in comment.lower():
            # Try to parse depot coordinates
            import re
            numbers = re.findall(r'[\d.]+', comment)
            if len(numbers) >= 2:
                depot_x, depot_y = float(numbers[0]), float(numbers[1])
                break
    
    # Write output
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"// 3D CETSP - Converted from {Path(input_path).name}\n")
        f.write(f"// Strategy: {strategy}, Z range: [{z_min}, {z_max}]\n")
        f.write(f"// Format: x y z radius\n")
        f.write(f"// Depot is at ({depot_x}, {depot_y}, {depot_z})\n")
        f.write(f"// {len(nodes_3d)} customers\n")
        f.write("\n")
        
        # Write nodes
        for x, y, z, radius in nodes_3d:
            f.write(f"{x:.1f} {y:.1f} {z:.1f} {radius:.1f}\n")
    
    return output_path


def batch_convert(
    input_dir: str,
    output_dir: Optional[str] = None,
    strategy: Literal['wave', 'random', 'dome', 'layers', 'distance'] = 'wave',
    z_min: float = 10.0,
    z_max: float = 90.0,
    seed: Optional[int] = 42
) -> list[str]:
    """
    Convert all CETSP files in a directory to 3D.
    
    Args:
        input_dir: Directory containing 2D CETSP files
        output_dir: Output directory (created if doesn't exist)
        strategy: Z-coordinate generation strategy
        z_min: Minimum z value
        z_max: Maximum z value
        seed: Random seed
    
    Returns:
        List of output file paths
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path.parent / f"{input_path.name}_3D"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    
    for cetsp_file in input_path.glob("*.cetsp"):
        output_file = output_path / f"{cetsp_file.stem}_3d.cetsp"
        
        try:
            convert_cetsp_to_3d(
                str(cetsp_file),
                str(output_file),
                strategy=strategy,
                z_min=z_min,
                z_max=z_max,
                seed=seed
            )
            output_files.append(str(output_file))
            print(f"✓ Converted: {cetsp_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"✗ Failed: {cetsp_file.name} - {e}")
    
    return output_files


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 2D CETSP to 3D CETSP")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "-s", "--strategy", 
        choices=['wave', 'random', 'dome', 'layers', 'distance'],
        default='wave',
        help="Z-coordinate generation strategy"
    )
    parser.add_argument("--z-min", type=float, default=10.0, help="Minimum z value")
    parser.add_argument("--z-max", type=float, default=90.0, help="Maximum z value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch", action="store_true", help="Batch convert directory")
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        files = batch_convert(
            args.input, args.output,
            strategy=args.strategy,
            z_min=args.z_min,
            z_max=args.z_max,
            seed=args.seed
        )
        print(f"\nConverted {len(files)} files")
    else:
        output = convert_cetsp_to_3d(
            args.input, args.output,
            strategy=args.strategy,
            z_min=args.z_min,
            z_max=args.z_max,
            seed=args.seed
        )
        print(f"Created: {output}")
