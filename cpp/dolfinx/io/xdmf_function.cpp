// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_function.h"
#include "pugixml.hpp"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <string>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
/// String suffix for real and complex components of a vector-valued
/// field
constexpr std::array field_ext = {"real_", "imag_"};

// value_rank to the XDMF string description
constexpr std::array rank_to_string = {"Scalar", "Vector", "Tensor"};

//-----------------------------------------------------------------------------

/// Returns true for DG0 fem::Functions
bool has_cell_centred_data(const fem::FunctionSpace& V)
{
  int cell_based_dim = 1;
  const int rank = V.element()->value_shape().size();
  for (int i = 0; i < rank; i++)
    cell_based_dim *= V.mesh()->topology().dim();
  return (V.dofmap()->element_dof_layout().num_dofs() * V.dofmap()->bs()
          == cell_based_dim);
}
//-----------------------------------------------------------------------------

// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
int get_padded_width(const fem::FiniteElement& e)
{
  const int width = e.value_size();
  const int rank = e.value_shape().size();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
}
//-----------------------------------------------------------------------------
template <typename T>
void _add_function(MPI_Comm comm, const fem::Function<T>& u, const double t,
                   pugi::xml_node& xml_node, const hid_t h5_id)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  auto V = u.function_space();
  assert(V);
  std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
  assert(mesh);

  // Get fem::Function data values and shape
  std::vector<T> data_values;
  const bool cell_centred = has_cell_centred_data(*V);
  if (cell_centred)
    data_values = xdmf_utils::get_cell_data_values(u);
  else
    data_values = xdmf_utils::get_point_data_values(u);

  auto map_c = mesh->topology().index_map(mesh->topology().dim());
  assert(map_c);

  auto map_v = mesh->geometry().index_map();
  assert(map_v);

  // Add attribute DataItem node and write data
  const int width = get_padded_width(*V->element());
  assert(data_values.size() % width == 0);
  const int num_values
      = cell_centred ? map_c->size_global() : map_v->size_global();

  const int value_rank = V->element()->value_shape().size();

  std::string t_str = boost::lexical_cast<std::string>(t);
  std::replace(t_str.begin(), t_str.end(), '.', '_');

  const bool use_mpi_io = dolfinx::MPI::size(comm) > 1;
  // ---
  if constexpr (std::is_scalar_v<T>)
  {
    // -- Real
    std::string dataset_name
        = std::string("/Function/") + u.name + std::string("/") + t_str;

    // Add attribute node
    pugi::xml_node attribute_node = xml_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = u.name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string[value_rank];
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

    // Add data item
    const std::int64_t num_local = data_values.size() / width;
    std::int64_t offset = 0;
    MPI_Exscan(&num_local, &offset, 1, dolfinx::MPI::mpi_type<std::int64_t>(),
               MPI_SUM, comm);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name, data_values,
                              offset, {num_values, width}, "", use_mpi_io);
  }
  else
  {
    // -- Complex
    using U = typename T::value_type;

    std::array<std::vector<U>, 2> component_data_values{
        std::vector<U>(data_values.size()), std::vector<U>(data_values.size())};
    std::transform(data_values.cbegin(), data_values.cend(),
                   component_data_values[0].begin(),
                   [](auto& x) { return x.real(); });
    std::transform(data_values.cbegin(), data_values.cend(),
                   component_data_values[1].begin(),
                   [](auto& x) { return x.imag(); });

    const std::int64_t num_local = data_values.size() / width;
    std::int64_t offset = 0;
    MPI_Exscan(&num_local, &offset, 1, dolfinx::MPI::mpi_type<std::int64_t>(),
               MPI_SUM, comm);

    for (std::size_t i = 0; i < 2; ++i)
    {
      std::string attr_name = field_ext[i] + u.name;
      std::string dataset_name
          = std::string("/Function/") + attr_name + std::string("/") + t_str;

      // Add attribute node
      pugi::xml_node attribute_node = xml_node.append_child("Attribute");
      assert(attribute_node);
      attribute_node.append_attribute("Name") = attr_name.c_str();
      attribute_node.append_attribute("AttributeType")
          = rank_to_string[value_rank];
      attribute_node.append_attribute("Center")
          = cell_centred ? "Cell" : "Node";

      // Add data item of component
      xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name,
                                component_data_values[i], offset,
                                {num_values, width}, "", use_mpi_io);
    }
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
void xdmf_function::add_function(MPI_Comm comm, const fem::Function<double>& u,
                                 const double t, pugi::xml_node& xml_node,
                                 const hid_t h5_id)
{
  _add_function(comm, u, t, xml_node, h5_id);
}
//-----------------------------------------------------------------------------
void xdmf_function::add_function(MPI_Comm comm,
                                 const fem::Function<std::complex<double>>& u,
                                 const double t, pugi::xml_node& xml_node,
                                 const hid_t h5_id)
{
  _add_function(comm, u, t, xml_node, h5_id);
}
//-----------------------------------------------------------------------------
