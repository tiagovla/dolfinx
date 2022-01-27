// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <xtensor/xtensor.hpp>

namespace dolfinx::mesh
{
class Mesh;
}

namespace dolfinx::io
{
/// Extract the cell topology (connectivity) in VTK ordering for all
/// cells the mesh. The 'topology' includes higher-order 'nodes'. The
/// index of a 'node' corresponds to the index of DOLFINx geometry
/// 'nodes'.
/// @param [in] mesh The mesh
/// @return The cell topology in VTK ordering and in term of the DOLFINx
/// geometry 'nodes'
/// @note The indices in the return array correspond to the point
/// indices in the mesh geometry array
/// @note Even if the indices are local (int32), both Fides and VTX
/// require int64 as local input
xt::xtensor<std::int64_t, 2> extract_vtk_connectivity(const mesh::Mesh& mesh);

} // namespace dolfinx::io
