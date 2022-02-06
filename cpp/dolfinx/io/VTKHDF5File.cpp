// Copyright (C) 2005-2020 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKHDF5File.h"
#include "HDF5Interface.h"
#include "cells.h"
#include "vtk_utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <string>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx;

namespace
{

} // namespace

//----------------------------------------------------------------------------
io::VTKHDF5File::VTKHDF5File(MPI_Comm comm,
                             const std::filesystem::path& filename,
                             const std::string&)
    : _hdf5_handle(-1), _comm(comm)
{
  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

  MPI_Info info;
  MPI_Info_create(&info);
  if (H5Pset_fapl_mpio(plist_id, comm, info) < 0)
    throw std::runtime_error("Call to H5Pset_fapl_mpio unsuccessful");
  MPI_Info_free(&info);

  _hdf5_handle
      = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  if (_hdf5_handle < 0)
    throw std::runtime_error("Failed to create HDF5 file.");

  if (H5Pclose(plist_id) < 0)
    throw std::runtime_error("Failed to close HDF5 file property list.");

  // --
  std::string group_name = "VTKHDF";

  // // Prepend a slash if missing
  // if (_group_name[0] != '/')
  //   _group_name = "/" + _group_name;

  const hid_t group_id_vis = H5Gcreate2(_hdf5_handle, group_name.c_str(),
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (group_id_vis < 0)
    throw std::runtime_error("Failed to add HDF5 group");

  hsize_t shape= 2;
  hid_t dataspace_id = H5Screate_simple(1, &shape, NULL);
  hid_t attr = H5Acreate2(group_id_vis, "Version", H5T_STD_I64LE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT);

  std::array<std::int64_t, 2> version = {1, 0};
  herr_t status = H5Awrite(attr, H5T_STD_I64LE, version.data());

  // std::array<std::int64_t, 2> version = {1, 0};
  // herr_t status = H5Swrite(spaceid, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
  //                          H5P_DEFAULT, version.data());

  H5Aclose(attr);
  H5Sclose(dataspace_id);

  if (H5Gclose(group_id_vis) < 0)
    throw std::runtime_error("Failed to close HDF5 group");
}
//----------------------------------------------------------------------------
io::VTKHDF5File::~VTKHDF5File() { H5Fclose(_hdf5_handle); }
//----------------------------------------------------------------------------
// void io::VTKHDF5File::close()
// {
//   if (_hdf5_handle > 0 and H5Fclose(_hdf5_handle) < 0)
//     throw std::runtime_error("Failed to close HDF5 file.");
// }
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void io::VTKHDF5File::write(const mesh::Mesh& mesh, double time)
{
  // io::HDF5Interface::add_group(_hdf5_handle, "VTKHDF");

  // if (!_pvd_xml)
  //   throw std::runtime_error("VTKFile has already been closed");

  // // Get the PVD "Collection" node
  // pugi::xml_node xml_collections
  //     = _pvd_xml->child("VTKFile").child("Collection");
  // assert(xml_collections);

  // // Compute counter string
  // const std::string counter_str = get_counter(xml_collections, "DataSet");

  // // Get mesh data for this rank
  // const mesh::Topology& topology = mesh.topology();
  // const mesh::Geometry& geometry = mesh.geometry();
  // auto xmap = geometry.index_map();
  // assert(xmap);
  // const int tdim = topology.dim();
  // const std::int32_t num_points = xmap->size_local() + xmap->num_ghosts();
  // const std::int32_t num_cells = topology.index_map(tdim)->size_local()
  //                                + topology.index_map(tdim)->num_ghosts();

  // // Create a VTU XML object
  // pugi::xml_document xml_vtu;
  // pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  // vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  // vtk_node_vtu.append_attribute("version") = "2.2";
  // pugi::xml_node grid_node_vtu =
  // vtk_node_vtu.append_child("UnstructuredGrid");

  // // Add "Piece" node and required metadata
  // pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  // piece_node.append_attribute("NumberOfPoints") = num_points;
  // piece_node.append_attribute("NumberOfCells") = num_cells;

  // // Add mesh data to "Piece" node
  // xt::xtensor<std::int64_t, 2> cells = extract_vtk_connectivity(mesh);
  // xt::xtensor<double, 2> x
  //     = xt::adapt(geometry.x().data(), geometry.x().size(),
  //     xt::no_ownership(),
  //                 std::vector({geometry.x().size() / 3, std::size_t(3)}));
  // std::vector<std::uint8_t> x_ghost(x.shape(0), 0);
  // std::fill(std::next(x_ghost.begin(), xmap->size_local()), x_ghost.end(),
  // 1); add_mesh(x, geometry.input_global_indices(), x_ghost, cells,
  //          *topology.index_map(tdim), topology.cell_type(), topology.dim(),
  //          piece_node);

  // // Create filepath for a .vtu file
  // auto create_vtu_path = [file_root = _filename.parent_path(),
  //                         file_name = _filename.stem(), counter_str](int
  //                         rank)
  // {
  //   std::filesystem::path vtu = file_root / file_name;
  //   vtu += +"_p" + std::to_string(rank) + "_" + counter_str;
  //   vtu.replace_extension("vtu");
  //   return vtu;
  // };

  // // Save VTU XML to file
  // const int mpi_rank = dolfinx::MPI::rank(_comm.comm());
  // std::filesystem::path vtu = create_vtu_path(mpi_rank);
  // if (vtu.has_parent_path())
  //   std::filesystem::create_directories(vtu.parent_path());
  // xml_vtu.save_file(vtu.c_str(), "  ");

  // // Create a PVTU XML object on rank 0
  // std::filesystem::path p_pvtu = _filename.stem();
  // p_pvtu += counter_str;
  // p_pvtu.replace_extension("pvtu");
  // if (mpi_rank == 0)
  // {
  //   pugi::xml_document xml_pvtu;
  //   pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
  //   vtk_node.append_attribute("type") = "PUnstructuredGrid";
  //   vtk_node.append_attribute("version") = "1.0";
  //   pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  //   grid_node.append_attribute("GhostLevel") = 1;

  //   // Add mesh metadata to PVTU object
  //   add_pvtu_mesh(grid_node);

  //   // Add data for each process to the PVTU object
  //   const int mpi_size = MPI::size(_comm.comm());
  //   for (int r = 0; r < mpi_size; ++r)
  //   {
  //     std::filesystem::path vtu = create_vtu_path(r);
  //     pugi::xml_node piece_node = grid_node.append_child("Piece");
  //     piece_node.append_attribute("Source") = vtu.filename().c_str();
  //   }

  //   // Write PVTU file
  //   if (p_pvtu.has_parent_path())
  //     std::filesystem::create_directories(p_pvtu.parent_path());
  //   xml_pvtu.save_file(p_pvtu.c_str(), "  ");

  //   // Append PVD file
  //   pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  //   dataset_node.append_attribute("timestep") = time;
  //   dataset_node.append_attribute("part") = "0";
  //   dataset_node.append_attribute("file") = p_pvtu.filename().c_str();
  // }
}
// //----------------------------------------------------------------------------
// void io::VTKFile::write_functions(
//     const std::vector<std::reference_wrapper<const fem::Function<double>>>&
//     u, double time)
// {
//   write_function(u, time, _pvd_xml, _filename);
// }
// //----------------------------------------------------------------------------
// void io::VTKFile::write_functions(
//     const std::vector<
//         std::reference_wrapper<const fem::Function<std::complex<double>>>>&
//         u,
//     double time)
// {
//   write_function(u, time, _pvd_xml, _filename);
// }
// //----------------------------------------------------------------------------
