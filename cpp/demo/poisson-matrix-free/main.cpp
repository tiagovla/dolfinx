

#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
/// Compute vector r = alpha*x + y
/// @param r Result
/// @param alpha
/// @param x
/// @param y
template <typename T>
void axpy(la::Vector<T>& r, T alpha, const la::Vector<T>& x,
          const la::Vector<T>& y)
{
  std::transform(x.array().cbegin(), x.array().cbegin() + x.map()->size_local(),
                 y.array().cbegin(), r.mutable_array().begin(),
                 [&alpha](const T& vx, const T& vy)
                 { return vx * alpha + vy; });
}

/// Solve problem A.x = b with Conjugate Gradient method
/// @param b RHS Vector
/// @param x Solution Vector
/// @param matvec_function Function that provides the operator action
/// @param kmax Maxmimum number of iterations
template <typename T>
int cg(la::Vector<T>& x, const la::Vector<T>& b,
       std::function<void(la::Vector<T>&, la::Vector<T>&)> matvec_function,
       int kmax = 50, double rtol = 1e-8)
{
  int M = b.map()->size_local();

  // Residual vector
  la::Vector<T> r(b); // or b - A.x0
  la::Vector<T> y(b);
  la::Vector<T> p(x);
  std::copy(r.array().begin(), r.array().begin() + M,
            p.mutable_array().begin());

  double rnorm0 = r.squared_norm();

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // MatVec
    // y = A.p;
    matvec_function(p, y);

    // Calculate alpha = r.r/p.y
    const double alpha = rnorm / la::inner_product(p, y);

    // Update x and r
    // x = x + alpha*p
    axpy(x, alpha, p, x);

    // r = r - alpha*y
    axpy(r, -alpha, y, r);

    // Update rnorm
    const double rnorm_new = r.squared_norm();
    const double beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    std::cout << "it " << k << ": " << std::sqrt(rnorm / rnorm0) << std::endl;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p.
    // p = beta*p + r
    axpy(p, beta, p, r);
  }
  return k;
}
} // namespace

using T = PetscScalar;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {32, 16},
        mesh::CellType::triangle, mesh::GhostMode::none));

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Next, we define the variational formulation by initializing the
    // bilinear and linear forms (:math:`a`, :math:`L`) using the previously
    // defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
    // source and boundary flux term (:math:`f`, :math:`g`) and attach these
    // to the linear form.
    //
    // .. code-block:: cpp

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

    // Now, the Dirichlet boundary condition (:math:`u = 0`) can be created
    // using the class :cpp:class:`DirichletBC`. A :cpp:class:`DirichletBC`
    // takes two arguments: the value of the boundary condition,
    // and the part of the boundary on which the condition applies.
    // In our example, the value of the boundary condition (0.0) can
    // represented using a :cpp:class:`Function`, and the Dirichlet boundary
    // is defined by the indices of degrees of freedom to which the boundary
    // condition applies.
    // The definition of the Dirichlet boundary condition then looks
    // as follows:
    //
    // .. code-block:: cpp

    // Define boundary condition

    auto facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto& x) -> xt::xtensor<bool, 1>
        {
          auto x0 = xt::row(x, 0);
          return xt::isclose(x0, 0.0) or xt::isclose(x0, 2.0);
        });
    const auto bdofs = fem::locate_dofs_topological({*V}, 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    f->interpolate(
        [](auto& x) -> xt::xarray<T>
        {
          auto dx = xt::square(xt::row(x, 0) - 0.5)
                    + xt::square(xt::row(x, 1) - 0.5);
          return 10 * xt::exp(-(dx) / 0.02);
        });

    g->interpolate(
        [](auto& x) -> xt::xarray<T> { return xt::sin(5 * xt::row(x, 0)); });

    // Compute solution
    fem::Function<T> u(V);
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                      {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting(b.mutable_array(), {a}, {{bc}}, {}, 1.0);
    b.scatter_rev(common::IndexMap::Mode::add);
    fem::set_bc(b.mutable_array(), {bc});

    auto&& dof_ghosts = V->dofmap()->index_map->ghosts();
    std::vector<PetscInt> ghosts(dof_ghosts.begin(), dof_ghosts.end());

    std::function<void(la::Vector<T>&, la::Vector<T>&)> matvec
        = [&ghosts, &A](la::Vector<T>& x, la::Vector<T>& y)
    {
      const PetscInt local_size = x.map()->size_local();
      const PetscInt global_size = x.map()->size_global();
      const PetscInt num_ghosts = ghosts.size();
      const PetscInt* ghosts_ptr = ghosts.data();

      MPI_Comm comm = x.map()->comm(common::IndexMap::Direction::forward);

      Vec _x_petsc = nullptr;
      Vec _y_petsc = nullptr;

      // Creates a parallel vector with ghost padding on each processor
      VecCreateGhostWithArray(comm, local_size, global_size, num_ghosts,
                              ghosts_ptr, x.array().data(), &_x_petsc);
      VecCreateGhostWithArray(comm, local_size, global_size, num_ghosts,
                              ghosts_ptr, y.mutable_array().data(), &_y_petsc);

      // Actual matrix vector multiplication
      MatMult(A.mat(), _x_petsc, _y_petsc);
    };

    cg(*u.x(), b, matvec, 100, 1e-6);

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write({u}, 0.0);
  }

  common::subsystem::finalize_petsc();
  return 0;
}
