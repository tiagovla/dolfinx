// Copyright (C) 2019-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::graph
{

/// Construct adjacency list data for a problem with a fixed number of
/// links (edges) for each node
/// @param [in] array Two-dimensional array of adjacency data where
/// matrix(i, j) is the jth neighbor of the ith node
/// @return Adjacency list data and offset array
template <typename T>
auto create_adjacency_data(const xt::xtensor<T, 2>& array)
{
  std::vector<T> data(array.shape(0) * array.shape(1));
  std::vector<std::int32_t> offset(array.shape(0) + 1, 0);
  for (std::size_t i = 0; i < array.shape(0); ++i)
  {
    for (std::size_t j = 0; j < array.shape(1); ++j)
      data[i * array.shape(1) + j] = array(i, j);
    offset[i + 1] = offset[i] + array.shape(1);
  }
  return std::pair(std::move(data), std::move(offset));
}

/// This class provides a static adjacency list data structure. It is
/// commonly used to store directed graphs. For each node in the
/// contiguous list of nodes [0, 1, 2, ..., n) it stores the connected
/// nodes. The representation is strictly local, i.e. it is not parallel
/// aware.
template <typename T>
class AdjacencyList
{
public:
  /// Construct trivial adjacency list where each of the n nodes is
  /// connected to itself
  /// @param [in] n Number of nodes
  explicit AdjacencyList(const std::int32_t n) : _array(n), _offsets(n + 1)
  {
    std::iota(_array.begin(), _array.end(), 0);
    std::iota(_offsets.begin(), _offsets.end(), 0);
  }

  /// Construct adjacency list from arrays of data
  /// @param [in] data Adjacency array
  /// @param [in] offsets The index to the adjacency list in the data
  /// array for node i
  template <
      typename U, typename V,
      typename = std::enable_if_t<
          std::is_same<std::vector<T>, std::decay_t<U>>::value
          && std::is_same<std::vector<std::int32_t>, std::decay_t<V>>::value>>
  AdjacencyList(U&& data, V&& offsets)
      : _array(std::forward<U>(data)), _offsets(std::forward<V>(offsets))
  {
    _array.reserve(_offsets.back());
    assert(_offsets.back() == (std::int32_t)_array.size());
  }

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a `std::vector<<std::vector<std::size_t>>`,
  /// `std::vector<<std::set<std::size_t>>`, etc).
  /// @param [in] data Adjacency list data, where `std::next(data, i)`
  /// points to the container of edges for node `i`.
  template <typename X>
  explicit AdjacencyList(const std::vector<X>& data)
  {
    // Initialize offsets and compute total size
    _offsets.reserve(data.size() + 1);
    _offsets.push_back(0);
    for (auto row = data.begin(); row != data.end(); ++row)
      _offsets.push_back(_offsets.back() + row->size());

    _array.reserve(_offsets.back());
    for (auto e = data.begin(); e != data.end(); ++e)
      _array.insert(_array.end(), e->begin(), e->end());
  }

  /// Copy constructor
  AdjacencyList(const AdjacencyList& list) = default;

  /// Move constructor
  AdjacencyList(AdjacencyList&& list) = default;

  /// Destructor
  ~AdjacencyList() = default;

  /// Assignment operator
  AdjacencyList& operator=(const AdjacencyList& list) = default;

  /// Move assignment operator
  AdjacencyList& operator=(AdjacencyList&& list) = default;

  /// Equality operator
  /// @return True is the adjacency lists are equal
  bool operator==(const AdjacencyList& list) const
  {
    return this->_array == list._array and this->_offsets == list._offsets;
  }

  /// Iterator
  template <typename V>
  struct Iterator
  {

    /// Type
    using X = typename std::conditional<
        std::is_const<V>::value,
        typename std::add_const<typename V::value_type>::type,
        typename std::remove_const<typename V::value_type>::type>::type;
    /// X
    using iterator_category = std::forward_iterator_tag; /// X
    /// X
    using difference_type = std::ptrdiff_t; /// X
    /// X
    using value_type = xtl::span<X>; /// X
    /// X
    using pointer = xtl::span<X>*; /// or also value_type*
    /// X
    using reference = xtl::span<X>&; /// or also value_type&

    Iterator(V& array, const std::vector<std::int32_t>& offsets,
             std::vector<std::int32_t>::const_iterator it)
        : _array(array), _offsets(offsets), _it(it)
    {
      if (std::next(_it) != _offsets.end())
        _row = xtl::span<X>(_array.data() + *_it, *(_it + 1) - *_it);
    }

    /// X
    difference_type distance(const Iterator& first, const Iterator& last) const
    {
      return std::distance(first._it, last._it);
    }

    /// X
    reference operator*() { return _row; }

    /// X
    pointer operator->() { return &_row; }

    /// X
    Iterator& operator++()
    {
      ++_it;
      if (std::next(_it) != _offsets.end())
        _row = xtl::span<X>(_array.data() + *_it, *(_it + 1) - *_it);
      else
        _row = xtl::span<X>();

      return *this;
    }

    // Iterator operator++(int)
    // {
    //   Iterator tmp = *this;
    //   ++(*this);
    //   return tmp;
    // }

    /// X
    friend bool operator==(const Iterator& a, const Iterator& b)
    {
      return a._row.data() == b._row.data() and a._row.size() == b._row.size();
    }

    /// X
    friend bool operator!=(const Iterator& a, const Iterator& b)
    {
      return a._row.data() != b._row.data() or a._row.size() != b._row.size();
    }

  private:
    V& _array;
    const std::vector<std::int32_t>& _offsets;
    std::vector<std::int32_t>::const_iterator _it;

    xtl::span<X> _row;
  };

  /// Begin iterator
  auto begin()
  {
    return Iterator<std::vector<T>>(_array, _offsets, _offsets.begin());
  }

  /// End iterator
  auto end()
  {
    return Iterator<std::vector<T>>(_array, _offsets,
                                    std::prev(_offsets.end()));
  }

  /// Begin iterator (const)
  auto begin() const
  {
    return Iterator<const std::vector<T>>(_array, _offsets, _offsets.begin());
  }

  /// End iterator (const)
  auto end() const
  {
    return Iterator<const std::vector<T>>(_array, _offsets,
                                          std::prev(_offsets.end()));
  }

  /// Get the number of nodes
  /// @return The number of nodes in the adjacency list
  std::size_t num_nodes() const { return _offsets.size() - 1; }

  /// Number of connections for given node
  /// @param [in] node Node index
  /// @return The number of outgoing links (edges) from the node
  std::size_t num_links(int node) const
  {
    assert((node + 1) < (int)_offsets.size());
    return _offsets[node + 1] - _offsets[node];
  }

  /// Get the links (edges) for given node
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  /// AdjacencyList::num_links(node).
  xtl::span<T> links(int node)
  {
    return xtl::span<T>(_array.data() + _offsets[node],
                        _offsets[node + 1] - _offsets[node]);
  }

  /// Get the links (edges) for given node (const version)
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  /// AdjacencyList:num_links(node).
  xtl::span<const T> links(int node) const
  {
    return xtl::span<const T>(_array.data() + _offsets[node],
                              _offsets[node + 1] - _offsets[node]);
  }

  /// Return contiguous array of links for all nodes (const version)
  const std::vector<T>& array() const { return _array; }

  /// Return contiguous array of links for all nodes
  std::vector<T>& array() { return _array; }

  /// Offset for each node in array() (const version)
  const std::vector<std::int32_t>& offsets() const { return _offsets; }

  /// Offset for each node in array()
  std::vector<std::int32_t>& offsets() { return _offsets; }

  /// Informal string representation (pretty-print)
  /// @return String representation of the adjacency list
  std::string str() const
  {
    std::stringstream s;
    s << "<AdjacencyList> with " + std::to_string(this->num_nodes()) + " nodes"
      << std::endl;
    for (std::size_t e = 0; e < _offsets.size() - 1; ++e)
    {
      s << "  " << e << ": [";
      for (auto link : this->links(e))
        s << link << " ";
      s << "]" << std::endl;
    }
    return s.str();
  }

private:
  // Connections for all entities stored as a contiguous array
  std::vector<T> _array;

  // Position of first connection for each entity (using local index)
  std::vector<std::int32_t> _offsets;
};

/// @brief Construct a constant degree (valency) adjacency list.
///
/// A constant degree graph has the same number of edges for every node.
/// @param [in] data Adjacency array
/// @param [in] degree The number of (outgoing) edges for each node
/// @return An adjacency list
template <typename U>
AdjacencyList<typename U::value_type> regular_adjacency_list(U&& data,
                                                             int degree)
{
  if (degree == 0 and !data.empty())
  {
    throw std::runtime_error("Degree is zero but data is not empty for "
                             "constant degree AdjacencyList");
  }

  if (degree > 0 and data.size() % degree != 0)
  {
    throw std::runtime_error(
        "Incompatible data size and degree for constant degree AdjacencyList");
  }

  std::int32_t num_nodes = degree == 0 ? data.size() : data.size() / degree;
  std::vector<std::int32_t> offsets(num_nodes + 1, 0);
  for (std::size_t i = 1; i < offsets.size(); ++i)
    offsets[i] = offsets[i - 1] + degree;
  return AdjacencyList<typename U::value_type>(std::forward<U>(data),
                                               std::move(offsets));
}

} // namespace dolfinx::graph
