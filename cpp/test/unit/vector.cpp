// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::Vector

#include <catch.hpp>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/Matrix.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>

// #include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

using namespace dolfinx;

namespace
{

void test_vector()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  const std::vector<int> global_ghost_owner(ghosts.size(),
                                            (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  const auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local,
      dolfinx::MPI::compute_graph_edges(
          MPI_COMM_WORLD,
          std::set<int>(global_ghost_owner.begin(), global_ghost_owner.end())),
      ghosts, global_ghost_owner);

  la::Vector<PetscScalar> v(index_map, 1);
  std::fill(v.mutable_array().begin(), v.mutable_array().end(), 1.0);

  const double norm2 = v.squared_norm();
  CHECK(norm2 == mpi_size * size_local);

  std::fill(v.mutable_array().begin(), v.mutable_array().end(), mpi_rank);

  const double sumn2
      = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(v.squared_norm() == sumn2);
  CHECK(v.norm(la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v, v) == sumn2);
  CHECK(v.norm(la::Norm::linf) == static_cast<PetscScalar>(mpi_size - 1));
}

void test_matrix()
{
  auto map0 = std::make_shared<common::IndexMap>(MPI_COMM_SELF, 8);
  la::SparsityPattern p(MPI_COMM_SELF, {map0, map0}, {1, 1});
  p.insert(std::vector{0}, std::vector{0});
  p.insert(std::vector{4}, std::vector{5});
  p.insert(std::vector{5}, std::vector{4});
  p.assemble();

  la::Matrix<float> A(p);
  A.add(std::vector<decltype(A)::value_type>{1}, std::vector{0},
        std::vector{0});
  A.add(std::vector<decltype(A)::value_type>{2.3}, std::vector{4},
        std::vector{5});

  const auto Adense = A.to_dense();
  xt::xtensor<float, 2> Aref = xt::zeros<float>({8, 8});
  Aref(0, 0) = 1;
  Aref(4, 5) = 2.3;
  CHECK((Adense == Aref));

  Aref(4, 4) = 2.3;
  CHECK((Adense != Aref));

  // const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  // const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  // constexpr int size_local = 100;

  // // Create some ghost entries on next process
  // int num_ghosts = (mpi_size - 1) * 3;
  // std::vector<std::int64_t> ghosts(num_ghosts);
  // for (int i = 0; i < num_ghosts; ++i)
  //   ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  // const std::vector<int> global_ghost_owner(ghosts.size(),
  //                                           (mpi_rank + 1) % mpi_size);

  // // Create an IndexMap
  // const auto index_map = std::make_shared<common::IndexMap>(
  //     MPI_COMM_WORLD, size_local,
  //     dolfinx::MPI::compute_graph_edges(
  //         MPI_COMM_WORLD,
  //         std::set<int>(global_ghost_owner.begin(),
  //         global_ghost_owner.end())),
  //     ghosts, global_ghost_owner);

  // la::Vector<PetscScalar> v(index_map, 1);
  // std::fill(v.mutable_array().begin(), v.mutable_array().end(), 1.0);

  // const double norm2 = v.squared_norm();
  // CHECK(norm2 == mpi_size * size_local);

  // std::fill(v.mutable_array().begin(), v.mutable_array().end(), mpi_rank);

  // const double sumn2
  //     = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  // CHECK(v.squared_norm() == sumn2);
  // CHECK(v.norm(la::Norm::l2) == std::sqrt(sumn2));
  // CHECK(la::inner_product(v, v) == sumn2);
  // CHECK(v.norm(la::Norm::linf) == static_cast<PetscScalar>(mpi_size - 1));
}

} // namespace

TEST_CASE("Linear Algebra Vector", "[la_vector]")
{
  CHECK_NOTHROW(test_vector());
  CHECK_NOTHROW(test_matrix());
}
