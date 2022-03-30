// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::MatrixCSR

#include <catch2/catch.hpp>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

using namespace dolfinx;

namespace
{

void test_range_iterator()
{
  std::vector<std::int64_t> data = {0, 1, 3, 0, 4, 3, 7, 4};
  const graph::AdjacencyList<std::int64_t> g
      = graph::regular_adjacency_list(std::move(data), 2);
  CHECK(g.num_nodes() == 4);

  {
    std::size_t count = 0;
    for (auto n : g)
    {
      CHECK(n.size() == 2);
      ++count;
    }
    CHECK(count == g.num_nodes());
  }

  {
    std::size_t count = 0;
    std::size_t d = 0;
    for (auto it = g.begin(); it != g.end(); ++it)
    {
      d += std::distance(g.begin(), it);
      CHECK(it->size() == 2);
      ++count;
    }
    CHECK(d == (g.num_nodes() - 1) * g.num_nodes() / 2);
    CHECK(count == g.num_nodes());
  }
}

} // namespace

TEST_CASE("Adjacency lists", "[adjacencylist]")
{
  CHECK_NOTHROW(test_range_iterator());
}