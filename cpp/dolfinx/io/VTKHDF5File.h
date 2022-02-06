// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <filesystem>
#include <functional>
#include <hdf5.h>
#include <memory>
#include <string>

namespace dolfinx::fem
{
template <typename T>
class Function;
}

namespace dolfinx::mesh
{
class Mesh;
}

namespace dolfinx::io
{

class VTKHDF5File
{
public:
  /// Create VTK file
  VTKHDF5File(MPI_Comm comm, const std::filesystem::path& filename,
              const std::string& file_mode);

  /// Destructor
  ~VTKHDF5File();

  /// Flushes XML files to disk
  void flush();

  /// Write a mesh to file. Supports arbitrary order Lagrange
  /// isoparametric cells.
  /// @param[in] mesh The Mesh to write to file
  /// @param[in] time Time parameter to associate with @p mesh
  void write(const mesh::Mesh& mesh, double time = 0.0);

  /// Write finite elements function with an associated timestep
  /// @param[in] u List of functions to write to file
  /// @param[in] t Time parameter to associate with @p u
  /// @pre All Functions in `u` must share the same mesh
  /// @pre All Functions in `u` with point-wise data must use the same
  /// element type (up to the block size) and the element must be
  /// (discontinuous) Lagrange
  /// @pre Functions in `u` cannot be sub-Functions. Interpolate
  /// sub-Functions before output
  template <typename T>
  void
  write(const std::vector<std::reference_wrapper<const fem::Function<T>>>& u,
        double t)
  {
    write_functions(u, t);
  }

private:
  void write_functions(
      const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
      double t);
  void write_functions(
      const std::vector<
          std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
      double t);

  hid_t _hdf5_handle;
  // std::filesystem::path _filename;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};
} // namespace dolfinx::io
