"""Test `zvtk` library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import zvtk

if TYPE_CHECKING:
    from pathlib import Path

    from pyvista.core.dataset import DataSet
    from pyvista.core.grid import ImageData
    from pyvista.core.grid import RectilinearGrid
    from pyvista.core.pointset import PointSet
    from pyvista.core.pointset import PolyData
    from pyvista.core.pointset import UnstructuredGrid


def populate_data(ds: DataSet) -> None:
    """Supply point, cell, and field data to any dataset."""
    rng = np.random.default_rng()

    # point data
    n_points = ds.n_points

    pd = ds.point_data
    pd["int8_data"] = rng.integers(-128, 127, n_points, dtype=np.int8)
    pd["int16_data"] = rng.integers(-32768, 32767, n_points, dtype=np.int16)
    pd["int32_data"] = rng.integers(-2147483648, 2147483647, n_points, dtype=np.int32)
    pd["int64_data"] = rng.integers(
        -9223372036854775808, 9223372036854775807, n_points, dtype=np.int64
    )
    pd["uint8_data"] = rng.integers(0, 255, n_points, dtype=np.uint8)
    pd["uint16_data"] = rng.integers(0, 65535, n_points, dtype=np.uint16)
    pd["uint32_data"] = rng.integers(0, 4294967295, n_points, dtype=np.uint32)
    pd["uint64_data"] = rng.integers(0, 18446744073709551615, n_points, dtype=np.uint64)
    pd["float32_data"] = rng.random(n_points, dtype=np.float32)
    pd["float64_data"] = rng.random(n_points, dtype=np.float64)

    # cell data
    n_cells = ds.n_cells
    if n_cells:
        cd = ds.cell_data
        cd["int32_data"] = rng.integers(-2147483648, 2147483647, n_cells, dtype=np.int32)
        cd["float64_data"] = rng.random(n_cells, dtype=np.float64)

    # point data
    scalars_name = "scalars"
    ds.point_data.set_array(np.arange(ds.n_points), scalars_name)
    ds.point_data.active_scalars_name = scalars_name

    vectors_name = "vectors"
    ds.point_data.set_array(rng.random((ds.n_points, 3)), vectors_name)
    ds.point_data.active_vectors_name = vectors_name

    textures_name = "textures"
    ds.point_data.set_array(rng.random((ds.n_points, 2)), textures_name)
    ds.point_data.active_texture_coordinates_name = textures_name

    normals_name = "normals"
    ds.point_data.set_array(rng.random((ds.n_points, 3)), normals_name)
    ds.point_data.active_normals_name = normals_name

    # cell data
    if n_cells:
        scalars_name = "scalars"
        ds.cell_data.set_array(np.arange(ds.n_cells), scalars_name)
        ds.cell_data.active_scalars_name = scalars_name

        vectors_name = "vectors"
        ds.cell_data.set_array(rng.random((ds.n_cells, 3)), vectors_name)
        ds.cell_data.active_vectors_name = vectors_name

        textures_name = "textures"
        ds.cell_data.set_array(rng.random((ds.n_cells, 2)), textures_name)
        ds.cell_data.active_texture_coordinates_name = textures_name

        normals_name = "normals"
        ds.cell_data.set_array(rng.random((ds.n_cells, 3)), normals_name)
        ds.cell_data.active_normals_name = normals_name

    # field data
    n = 10  # arbitrary size for field data
    ds.field_data["int32_field"] = np.arange(n, dtype=np.int32)
    ds.field_data["float64_field"] = rng.random(n, dtype=np.float64)
    ds.field_data["uint8_field"] = rng.integers(0, 255, n, dtype=np.uint8)


def test_ugrid(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test unstructured grid."""
    populate_data(ugrid)

    tmp_filename = tmp_path / "ugrid.zvtk"
    zvtk.write(ugrid, tmp_filename)
    ugrid_out = zvtk.read(tmp_filename)

    assert ugrid.point_data == ugrid_out.point_data
    assert ugrid.cell_data == ugrid_out.cell_data
    assert ugrid.field_data == ugrid_out.field_data

    # can remove with pyvista >= 0.47.0
    np.allclose(ugrid.polyhedron_faces, ugrid_out.polyhedron_faces)
    np.allclose(ugrid.polyhedron_face_locations, ugrid_out.polyhedron_face_locations)

    assert ugrid == ugrid_out


@pytest.mark.parametrize("strip", [False, True])
def test_polydata(polydata: PolyData, tmp_path: Path, *, strip: bool) -> None:
    """Test unstructured grid."""
    # add in separate lines and vertices
    polydata.lines = [2, 0, 1, 2, 1, 2, 2, 8, 9]
    polydata.verts = [1, 0, 1, 1, 1, 2, 1, 3, 1, 4]
    if strip:
        polydata = polydata.strip()

    populate_data(polydata)

    tmp_filename = tmp_path / "polydata.zvtk"
    zvtk.write(polydata, tmp_filename)
    polydata_out = zvtk.read(tmp_filename)

    assert polydata.n_cells == polydata_out.n_cells
    assert polydata.n_strips == polydata_out.n_strips
    assert polydata.n_points == polydata_out.n_points

    assert polydata.point_data == polydata_out.point_data
    assert polydata.cell_data == polydata_out.cell_data
    assert polydata.field_data == polydata_out.field_data
    assert polydata == polydata_out


def test_imagedata(imagedata: ImageData, tmp_path: Path) -> None:
    """Test unstructured grid."""
    populate_data(imagedata)

    tmp_filename = tmp_path / "imagedata.zvtk"
    zvtk.write(imagedata, tmp_filename)
    imagedata_out = zvtk.read(tmp_filename)

    assert imagedata.n_cells == imagedata_out.n_cells
    assert imagedata.n_points == imagedata_out.n_points

    assert imagedata.dimensions == imagedata_out.dimensions
    assert imagedata.spacing == imagedata_out.spacing
    assert imagedata.origin == imagedata_out.origin
    assert np.allclose(imagedata.direction_matrix, imagedata_out.direction_matrix)
    assert imagedata.offset == imagedata_out.offset

    assert imagedata.point_data == imagedata_out.point_data
    assert imagedata.cell_data == imagedata_out.cell_data
    assert imagedata.field_data == imagedata_out.field_data
    assert imagedata == imagedata_out


def test_pointset(pointset: PointSet, tmp_path: Path) -> None:
    """Test compressing a pointset."""
    populate_data(pointset)

    tmp_filename = tmp_path / "pointset.zvtk"
    zvtk.write(pointset, tmp_filename)
    pointset_out = zvtk.read(tmp_filename)

    assert pointset.n_points == pointset_out.n_points

    assert pointset.point_data == pointset_out.point_data
    assert pointset.field_data == pointset_out.field_data
    assert pointset == pointset_out


def test_rectilineargrid(rgrid: RectilinearGrid, tmp_path: Path) -> None:
    """Test compressing a RectilinearGrid."""
    populate_data(rgrid)

    tmp_filename = tmp_path / "rgrid.zvtk"
    zvtk.write(rgrid, tmp_filename)
    rgrid_out = zvtk.read(tmp_filename)

    assert rgrid.dimensions == rgrid_out.dimensions
    assert rgrid.n_points == rgrid_out.n_points
    assert rgrid.n_cells == rgrid_out.n_cells

    assert np.array_equal(rgrid.x, rgrid_out.x)
    assert np.array_equal(rgrid.y, rgrid_out.y)
    assert np.array_equal(rgrid.z, rgrid_out.z)

    assert rgrid.point_data == rgrid_out.point_data
    assert rgrid.cell_data == rgrid_out.cell_data
    assert rgrid.field_data == rgrid_out.field_data
    assert rgrid == rgrid_out
