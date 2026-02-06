"""Test cases for CETSP 2D to 3D converter."""

from pathlib import Path

import pytest

from src.cetsp.convert_to_3d import (
    add_z_coordinate,
    batch_convert,
    convert_cetsp_to_3d,
    parse_2d_cetsp,
)


class TestParse2DCETSP:
    """Test cases for parsing 2D CETSP files."""

    def test_parse_basic_file(self, tmp_path):
        """Test parsing a basic 2D CETSP file."""
        content = """// Test file
// Depot is at (100, 100, 0)
50 55 0 10 12
60 70 0 8 15
30 40 0 12 10
"""
        file_path = tmp_path / "test.cetsp"
        file_path.write_text(content)

        nodes, metadata = parse_2d_cetsp(str(file_path))

        assert len(nodes) == 3
        assert nodes[0] == (50.0, 55.0, 10.0)  # x, y, radius
        assert nodes[1] == (60.0, 70.0, 8.0)
        assert nodes[2] == (30.0, 40.0, 12.0)
        assert len(metadata["comments"]) == 2

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        file_path = tmp_path / "empty.cetsp"
        file_path.write_text("")

        nodes, metadata = parse_2d_cetsp(str(file_path))

        assert len(nodes) == 0

    def test_parse_with_comments_only(self, tmp_path):
        """Test parsing file with only comments."""
        content = """// Comment 1
// Comment 2
"""
        file_path = tmp_path / "comments.cetsp"
        file_path.write_text(content)

        nodes, metadata = parse_2d_cetsp(str(file_path))

        assert len(nodes) == 0
        assert len(metadata["comments"]) == 2

    def test_parse_minimal_columns(self, tmp_path):
        """Test parsing file with minimal columns (x, y only)."""
        content = """10 20
30 40
"""
        file_path = tmp_path / "minimal.cetsp"
        file_path.write_text(content)

        nodes, metadata = parse_2d_cetsp(str(file_path))

        assert len(nodes) == 2
        # Default radius should be applied
        assert nodes[0][2] == 10.0


class TestAddZCoordinate:
    """Test cases for adding Z coordinates."""

    @pytest.fixture
    def sample_nodes(self):
        """Sample 2D nodes for testing."""
        return [
            (10.0, 10.0, 5.0),
            (50.0, 50.0, 8.0),
            (90.0, 90.0, 6.0),
            (30.0, 70.0, 10.0),
        ]

    def test_wave_strategy(self, sample_nodes):
        """Test wave z-coordinate strategy."""
        result = add_z_coordinate(sample_nodes, strategy="wave", z_min=10, z_max=90)

        assert len(result) == 4
        for _x, _y, z, radius in result:
            assert 10 <= z <= 90
            assert radius > 0

    def test_random_strategy(self, sample_nodes):
        """Test random z-coordinate strategy."""
        result = add_z_coordinate(sample_nodes, strategy="random", z_min=10, z_max=90, seed=42)

        assert len(result) == 4
        for _x, _y, z, _radius in result:
            assert 10 <= z <= 90

    def test_dome_strategy(self, sample_nodes):
        """Test dome z-coordinate strategy (center is highest)."""
        result = add_z_coordinate(sample_nodes, strategy="dome", z_min=10, z_max=90)

        # Center node (50, 50) should have highest z
        z_values = {(r[0], r[1]): r[2] for r in result}
        center_z = z_values[(50.0, 50.0)]

        # Center should be higher than corners
        assert center_z > z_values[(10.0, 10.0)]
        assert center_z > z_values[(90.0, 90.0)]

    def test_layers_strategy(self, sample_nodes):
        """Test layers z-coordinate strategy."""
        result = add_z_coordinate(sample_nodes, strategy="layers", z_min=10, z_max=90)

        assert len(result) == 4
        # Layers should create 3 distinct heights
        z_values = [r[2] for r in result]
        unique_z = set(z_values)
        assert len(unique_z) <= 3

    def test_distance_strategy(self, sample_nodes):
        """Test distance-based z-coordinate strategy."""
        result = add_z_coordinate(sample_nodes, strategy="distance", z_min=10, z_max=90)

        # Center should have lowest z (inverse of dome)
        z_values = {(r[0], r[1]): r[2] for r in result}
        center_z = z_values[(50.0, 50.0)]

        assert center_z < z_values[(10.0, 10.0)]
        assert center_z < z_values[(90.0, 90.0)]

    def test_seed_reproducibility(self, sample_nodes):
        """Test that same seed produces same results."""
        result1 = add_z_coordinate(sample_nodes, strategy="random", seed=42)
        result2 = add_z_coordinate(sample_nodes, strategy="random", seed=42)

        assert result1 == result2

    def test_different_seeds_different_results(self, sample_nodes):
        """Test that different seeds produce different results."""
        result1 = add_z_coordinate(sample_nodes, strategy="random", seed=42)
        result2 = add_z_coordinate(sample_nodes, strategy="random", seed=123)

        # Z values should differ
        z1 = [r[2] for r in result1]
        z2 = [r[2] for r in result2]
        assert z1 != z2


class TestConvertCETSPTo3D:
    """Test cases for full file conversion."""

    def test_basic_conversion(self, tmp_path):
        """Test basic 2D to 3D conversion."""
        input_content = """// Test CETSP
// Depot is at (100, 100, 0)
50 55 0 10 12
60 70 0 8 15
"""
        input_path = tmp_path / "input.cetsp"
        input_path.write_text(input_content)

        output_path = tmp_path / "output_3d.cetsp"
        result = convert_cetsp_to_3d(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()

        # Read and verify output
        output_content = output_path.read_text()
        assert "3D CETSP" in output_content
        assert "Strategy:" in output_content

    def test_auto_output_path(self, tmp_path):
        """Test automatic output path generation."""
        input_content = """50 55 0 10 12
60 70 0 8 15
"""
        input_path = tmp_path / "myfile.cetsp"
        input_path.write_text(input_content)

        result = convert_cetsp_to_3d(str(input_path))

        expected_output = tmp_path / "myfile_3d.cetsp"
        assert result == str(expected_output)
        assert expected_output.exists()

    def test_all_strategies(self, tmp_path):
        """Test conversion with all strategies."""
        input_content = """50 55 0 10 12
60 70 0 8 15
30 40 0 12 10
"""
        input_path = tmp_path / "test.cetsp"
        input_path.write_text(input_content)

        strategies = ["wave", "random", "dome", "layers", "distance"]

        for strategy in strategies:
            output_path = tmp_path / f"test_{strategy}_3d.cetsp"
            result = convert_cetsp_to_3d(str(input_path), str(output_path), strategy=strategy)
            assert Path(result).exists()

    def test_custom_z_range(self, tmp_path):
        """Test conversion with custom z range."""
        input_content = """50 55 0 10 12
"""
        input_path = tmp_path / "test.cetsp"
        input_path.write_text(input_content)

        output_path = tmp_path / "output_3d.cetsp"
        convert_cetsp_to_3d(str(input_path), str(output_path), z_min=100, z_max=200)

        content = output_path.read_text()
        # Verify z range is documented
        assert "100" in content and "200" in content

    def test_empty_file_raises_error(self, tmp_path):
        """Test that empty file raises ValueError."""
        input_path = tmp_path / "empty.cetsp"
        input_path.write_text("")

        with pytest.raises(ValueError, match="No valid nodes"):
            convert_cetsp_to_3d(str(input_path))


class TestBatchConvert:
    """Test cases for batch conversion."""

    def test_batch_convert_multiple_files(self, tmp_path):
        """Test batch conversion of multiple files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create multiple test files
        for i in range(3):
            content = f"""// File {i}
{10 * i} {20 * i} 0 5 10
{30 * i} {40 * i} 0 8 15
"""
            (input_dir / f"test{i}.cetsp").write_text(content)

        output_files = batch_convert(str(input_dir), seed=42)

        assert len(output_files) == 3
        for output_file in output_files:
            assert Path(output_file).exists()

    def test_batch_convert_custom_output_dir(self, tmp_path):
        """Test batch conversion with custom output directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "custom_output"

        (input_dir / "test.cetsp").write_text("50 50 0 10 12")

        output_files = batch_convert(str(input_dir), str(output_dir))

        assert len(output_files) == 1
        assert output_dir.exists()
        assert (output_dir / "test_3d.cetsp").exists()

    def test_batch_convert_empty_dir(self, tmp_path):
        """Test batch conversion of empty directory."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        output_files = batch_convert(str(input_dir))

        assert len(output_files) == 0

    def test_batch_convert_with_strategy(self, tmp_path):
        """Test batch conversion with specific strategy."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        (input_dir / "test.cetsp").write_text("50 50 0 10 12\n60 60 0 8 15")

        output_files = batch_convert(str(input_dir), strategy="dome")

        assert len(output_files) == 1
        content = Path(output_files[0]).read_text()
        assert "dome" in content.lower()


class TestIntegration:
    """Integration tests with real-like data."""

    def test_realistic_cetsp_conversion(self, tmp_path):
        """Test conversion with realistic CETSP data."""
        # Create realistic 2D CETSP content
        content = """// Bubbles test instance
// Depot is at (100, 100, 0)
// 10 customers
50 55 0 10 12
60 70 0 8 15
30 40 0 12 10
80 20 0 6 8
15 85 0 9 11
90 60 0 7 9
45 30 0 11 13
70 90 0 5 7
25 65 0 8 10
55 15 0 10 12
"""
        input_path = tmp_path / "bubbles_test.cetsp"
        input_path.write_text(content)

        output_path = tmp_path / "bubbles_test_3d.cetsp"
        result = convert_cetsp_to_3d(
            str(input_path), str(output_path), strategy="wave", z_min=10, z_max=90, depot_z=50
        )

        # Verify output
        assert Path(result).exists()
        output_content = output_path.read_text()

        # Should have 10 data lines (excluding comments)
        data_lines = [
            line
            for line in output_content.split("\n")
            if line.strip() and not line.startswith("//")
        ]
        assert len(data_lines) == 10

        # Each line should have 4 values
        for line in data_lines:
            parts = line.split()
            assert len(parts) == 4
            # All should be numeric
            float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
