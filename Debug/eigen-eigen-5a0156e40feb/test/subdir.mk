################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../eigen-eigen-5a0156e40feb/test/cuda_basic.cu 

CPP_SRCS += \
../eigen-eigen-5a0156e40feb/test/adjoint.cpp \
../eigen-eigen-5a0156e40feb/test/array.cpp \
../eigen-eigen-5a0156e40feb/test/array_for_matrix.cpp \
../eigen-eigen-5a0156e40feb/test/array_of_string.cpp \
../eigen-eigen-5a0156e40feb/test/array_replicate.cpp \
../eigen-eigen-5a0156e40feb/test/array_reverse.cpp \
../eigen-eigen-5a0156e40feb/test/bandmatrix.cpp \
../eigen-eigen-5a0156e40feb/test/basicstuff.cpp \
../eigen-eigen-5a0156e40feb/test/bdcsvd.cpp \
../eigen-eigen-5a0156e40feb/test/bicgstab.cpp \
../eigen-eigen-5a0156e40feb/test/block.cpp \
../eigen-eigen-5a0156e40feb/test/boostmultiprec.cpp \
../eigen-eigen-5a0156e40feb/test/bug1213.cpp \
../eigen-eigen-5a0156e40feb/test/bug1213_main.cpp \
../eigen-eigen-5a0156e40feb/test/cholesky.cpp \
../eigen-eigen-5a0156e40feb/test/cholmod_support.cpp \
../eigen-eigen-5a0156e40feb/test/commainitializer.cpp \
../eigen-eigen-5a0156e40feb/test/conjugate_gradient.cpp \
../eigen-eigen-5a0156e40feb/test/conservative_resize.cpp \
../eigen-eigen-5a0156e40feb/test/constructor.cpp \
../eigen-eigen-5a0156e40feb/test/corners.cpp \
../eigen-eigen-5a0156e40feb/test/ctorleak.cpp \
../eigen-eigen-5a0156e40feb/test/denseLM.cpp \
../eigen-eigen-5a0156e40feb/test/dense_storage.cpp \
../eigen-eigen-5a0156e40feb/test/determinant.cpp \
../eigen-eigen-5a0156e40feb/test/diagonal.cpp \
../eigen-eigen-5a0156e40feb/test/diagonalmatrices.cpp \
../eigen-eigen-5a0156e40feb/test/dontalign.cpp \
../eigen-eigen-5a0156e40feb/test/dynalloc.cpp \
../eigen-eigen-5a0156e40feb/test/eigen2support.cpp \
../eigen-eigen-5a0156e40feb/test/eigensolver_complex.cpp \
../eigen-eigen-5a0156e40feb/test/eigensolver_generalized_real.cpp \
../eigen-eigen-5a0156e40feb/test/eigensolver_generic.cpp \
../eigen-eigen-5a0156e40feb/test/eigensolver_selfadjoint.cpp \
../eigen-eigen-5a0156e40feb/test/evaluators.cpp \
../eigen-eigen-5a0156e40feb/test/exceptions.cpp \
../eigen-eigen-5a0156e40feb/test/fastmath.cpp \
../eigen-eigen-5a0156e40feb/test/first_aligned.cpp \
../eigen-eigen-5a0156e40feb/test/geo_alignedbox.cpp \
../eigen-eigen-5a0156e40feb/test/geo_eulerangles.cpp \
../eigen-eigen-5a0156e40feb/test/geo_homogeneous.cpp \
../eigen-eigen-5a0156e40feb/test/geo_hyperplane.cpp \
../eigen-eigen-5a0156e40feb/test/geo_orthomethods.cpp \
../eigen-eigen-5a0156e40feb/test/geo_parametrizedline.cpp \
../eigen-eigen-5a0156e40feb/test/geo_quaternion.cpp \
../eigen-eigen-5a0156e40feb/test/geo_transformations.cpp \
../eigen-eigen-5a0156e40feb/test/half_float.cpp \
../eigen-eigen-5a0156e40feb/test/hessenberg.cpp \
../eigen-eigen-5a0156e40feb/test/householder.cpp \
../eigen-eigen-5a0156e40feb/test/incomplete_cholesky.cpp \
../eigen-eigen-5a0156e40feb/test/inplace_decomposition.cpp \
../eigen-eigen-5a0156e40feb/test/integer_types.cpp \
../eigen-eigen-5a0156e40feb/test/inverse.cpp \
../eigen-eigen-5a0156e40feb/test/is_same_dense.cpp \
../eigen-eigen-5a0156e40feb/test/jacobi.cpp \
../eigen-eigen-5a0156e40feb/test/jacobisvd.cpp \
../eigen-eigen-5a0156e40feb/test/linearstructure.cpp \
../eigen-eigen-5a0156e40feb/test/lscg.cpp \
../eigen-eigen-5a0156e40feb/test/lu.cpp \
../eigen-eigen-5a0156e40feb/test/mapped_matrix.cpp \
../eigen-eigen-5a0156e40feb/test/mapstaticmethods.cpp \
../eigen-eigen-5a0156e40feb/test/mapstride.cpp \
../eigen-eigen-5a0156e40feb/test/meta.cpp \
../eigen-eigen-5a0156e40feb/test/metis_support.cpp \
../eigen-eigen-5a0156e40feb/test/miscmatrices.cpp \
../eigen-eigen-5a0156e40feb/test/mixingtypes.cpp \
../eigen-eigen-5a0156e40feb/test/mpl2only.cpp \
../eigen-eigen-5a0156e40feb/test/nesting_ops.cpp \
../eigen-eigen-5a0156e40feb/test/nomalloc.cpp \
../eigen-eigen-5a0156e40feb/test/nullary.cpp \
../eigen-eigen-5a0156e40feb/test/numext.cpp \
../eigen-eigen-5a0156e40feb/test/packetmath.cpp \
../eigen-eigen-5a0156e40feb/test/pardiso_support.cpp \
../eigen-eigen-5a0156e40feb/test/pastix_support.cpp \
../eigen-eigen-5a0156e40feb/test/permutationmatrices.cpp \
../eigen-eigen-5a0156e40feb/test/prec_inverse_4x4.cpp \
../eigen-eigen-5a0156e40feb/test/product_extra.cpp \
../eigen-eigen-5a0156e40feb/test/product_large.cpp \
../eigen-eigen-5a0156e40feb/test/product_mmtr.cpp \
../eigen-eigen-5a0156e40feb/test/product_notemporary.cpp \
../eigen-eigen-5a0156e40feb/test/product_selfadjoint.cpp \
../eigen-eigen-5a0156e40feb/test/product_small.cpp \
../eigen-eigen-5a0156e40feb/test/product_symm.cpp \
../eigen-eigen-5a0156e40feb/test/product_syrk.cpp \
../eigen-eigen-5a0156e40feb/test/product_trmm.cpp \
../eigen-eigen-5a0156e40feb/test/product_trmv.cpp \
../eigen-eigen-5a0156e40feb/test/product_trsolve.cpp \
../eigen-eigen-5a0156e40feb/test/qr.cpp \
../eigen-eigen-5a0156e40feb/test/qr_colpivoting.cpp \
../eigen-eigen-5a0156e40feb/test/qr_fullpivoting.cpp \
../eigen-eigen-5a0156e40feb/test/qtvector.cpp \
../eigen-eigen-5a0156e40feb/test/rand.cpp \
../eigen-eigen-5a0156e40feb/test/real_qz.cpp \
../eigen-eigen-5a0156e40feb/test/redux.cpp \
../eigen-eigen-5a0156e40feb/test/ref.cpp \
../eigen-eigen-5a0156e40feb/test/resize.cpp \
../eigen-eigen-5a0156e40feb/test/rvalue_types.cpp \
../eigen-eigen-5a0156e40feb/test/schur_complex.cpp \
../eigen-eigen-5a0156e40feb/test/schur_real.cpp \
../eigen-eigen-5a0156e40feb/test/selfadjoint.cpp \
../eigen-eigen-5a0156e40feb/test/simplicial_cholesky.cpp \
../eigen-eigen-5a0156e40feb/test/sizeof.cpp \
../eigen-eigen-5a0156e40feb/test/sizeoverflow.cpp \
../eigen-eigen-5a0156e40feb/test/smallvectors.cpp \
../eigen-eigen-5a0156e40feb/test/sparseLM.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_basic.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_block.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_permutations.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_product.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_ref.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_solvers.cpp \
../eigen-eigen-5a0156e40feb/test/sparse_vector.cpp \
../eigen-eigen-5a0156e40feb/test/sparselu.cpp \
../eigen-eigen-5a0156e40feb/test/sparseqr.cpp \
../eigen-eigen-5a0156e40feb/test/special_numbers.cpp \
../eigen-eigen-5a0156e40feb/test/spqr_support.cpp \
../eigen-eigen-5a0156e40feb/test/stable_norm.cpp \
../eigen-eigen-5a0156e40feb/test/stddeque.cpp \
../eigen-eigen-5a0156e40feb/test/stddeque_overload.cpp \
../eigen-eigen-5a0156e40feb/test/stdlist.cpp \
../eigen-eigen-5a0156e40feb/test/stdlist_overload.cpp \
../eigen-eigen-5a0156e40feb/test/stdvector.cpp \
../eigen-eigen-5a0156e40feb/test/stdvector_overload.cpp \
../eigen-eigen-5a0156e40feb/test/superlu_support.cpp \
../eigen-eigen-5a0156e40feb/test/swap.cpp \
../eigen-eigen-5a0156e40feb/test/triangular.cpp \
../eigen-eigen-5a0156e40feb/test/umeyama.cpp \
../eigen-eigen-5a0156e40feb/test/umfpack_support.cpp \
../eigen-eigen-5a0156e40feb/test/unalignedassert.cpp \
../eigen-eigen-5a0156e40feb/test/unalignedcount.cpp \
../eigen-eigen-5a0156e40feb/test/upperbidiagonalization.cpp \
../eigen-eigen-5a0156e40feb/test/vectorization_logic.cpp \
../eigen-eigen-5a0156e40feb/test/vectorwiseop.cpp \
../eigen-eigen-5a0156e40feb/test/visitor.cpp \
../eigen-eigen-5a0156e40feb/test/zerosized.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/test/adjoint.o \
./eigen-eigen-5a0156e40feb/test/array.o \
./eigen-eigen-5a0156e40feb/test/array_for_matrix.o \
./eigen-eigen-5a0156e40feb/test/array_of_string.o \
./eigen-eigen-5a0156e40feb/test/array_replicate.o \
./eigen-eigen-5a0156e40feb/test/array_reverse.o \
./eigen-eigen-5a0156e40feb/test/bandmatrix.o \
./eigen-eigen-5a0156e40feb/test/basicstuff.o \
./eigen-eigen-5a0156e40feb/test/bdcsvd.o \
./eigen-eigen-5a0156e40feb/test/bicgstab.o \
./eigen-eigen-5a0156e40feb/test/block.o \
./eigen-eigen-5a0156e40feb/test/boostmultiprec.o \
./eigen-eigen-5a0156e40feb/test/bug1213.o \
./eigen-eigen-5a0156e40feb/test/bug1213_main.o \
./eigen-eigen-5a0156e40feb/test/cholesky.o \
./eigen-eigen-5a0156e40feb/test/cholmod_support.o \
./eigen-eigen-5a0156e40feb/test/commainitializer.o \
./eigen-eigen-5a0156e40feb/test/conjugate_gradient.o \
./eigen-eigen-5a0156e40feb/test/conservative_resize.o \
./eigen-eigen-5a0156e40feb/test/constructor.o \
./eigen-eigen-5a0156e40feb/test/corners.o \
./eigen-eigen-5a0156e40feb/test/ctorleak.o \
./eigen-eigen-5a0156e40feb/test/cuda_basic.o \
./eigen-eigen-5a0156e40feb/test/denseLM.o \
./eigen-eigen-5a0156e40feb/test/dense_storage.o \
./eigen-eigen-5a0156e40feb/test/determinant.o \
./eigen-eigen-5a0156e40feb/test/diagonal.o \
./eigen-eigen-5a0156e40feb/test/diagonalmatrices.o \
./eigen-eigen-5a0156e40feb/test/dontalign.o \
./eigen-eigen-5a0156e40feb/test/dynalloc.o \
./eigen-eigen-5a0156e40feb/test/eigen2support.o \
./eigen-eigen-5a0156e40feb/test/eigensolver_complex.o \
./eigen-eigen-5a0156e40feb/test/eigensolver_generalized_real.o \
./eigen-eigen-5a0156e40feb/test/eigensolver_generic.o \
./eigen-eigen-5a0156e40feb/test/eigensolver_selfadjoint.o \
./eigen-eigen-5a0156e40feb/test/evaluators.o \
./eigen-eigen-5a0156e40feb/test/exceptions.o \
./eigen-eigen-5a0156e40feb/test/fastmath.o \
./eigen-eigen-5a0156e40feb/test/first_aligned.o \
./eigen-eigen-5a0156e40feb/test/geo_alignedbox.o \
./eigen-eigen-5a0156e40feb/test/geo_eulerangles.o \
./eigen-eigen-5a0156e40feb/test/geo_homogeneous.o \
./eigen-eigen-5a0156e40feb/test/geo_hyperplane.o \
./eigen-eigen-5a0156e40feb/test/geo_orthomethods.o \
./eigen-eigen-5a0156e40feb/test/geo_parametrizedline.o \
./eigen-eigen-5a0156e40feb/test/geo_quaternion.o \
./eigen-eigen-5a0156e40feb/test/geo_transformations.o \
./eigen-eigen-5a0156e40feb/test/half_float.o \
./eigen-eigen-5a0156e40feb/test/hessenberg.o \
./eigen-eigen-5a0156e40feb/test/householder.o \
./eigen-eigen-5a0156e40feb/test/incomplete_cholesky.o \
./eigen-eigen-5a0156e40feb/test/inplace_decomposition.o \
./eigen-eigen-5a0156e40feb/test/integer_types.o \
./eigen-eigen-5a0156e40feb/test/inverse.o \
./eigen-eigen-5a0156e40feb/test/is_same_dense.o \
./eigen-eigen-5a0156e40feb/test/jacobi.o \
./eigen-eigen-5a0156e40feb/test/jacobisvd.o \
./eigen-eigen-5a0156e40feb/test/linearstructure.o \
./eigen-eigen-5a0156e40feb/test/lscg.o \
./eigen-eigen-5a0156e40feb/test/lu.o \
./eigen-eigen-5a0156e40feb/test/mapped_matrix.o \
./eigen-eigen-5a0156e40feb/test/mapstaticmethods.o \
./eigen-eigen-5a0156e40feb/test/mapstride.o \
./eigen-eigen-5a0156e40feb/test/meta.o \
./eigen-eigen-5a0156e40feb/test/metis_support.o \
./eigen-eigen-5a0156e40feb/test/miscmatrices.o \
./eigen-eigen-5a0156e40feb/test/mixingtypes.o \
./eigen-eigen-5a0156e40feb/test/mpl2only.o \
./eigen-eigen-5a0156e40feb/test/nesting_ops.o \
./eigen-eigen-5a0156e40feb/test/nomalloc.o \
./eigen-eigen-5a0156e40feb/test/nullary.o \
./eigen-eigen-5a0156e40feb/test/numext.o \
./eigen-eigen-5a0156e40feb/test/packetmath.o \
./eigen-eigen-5a0156e40feb/test/pardiso_support.o \
./eigen-eigen-5a0156e40feb/test/pastix_support.o \
./eigen-eigen-5a0156e40feb/test/permutationmatrices.o \
./eigen-eigen-5a0156e40feb/test/prec_inverse_4x4.o \
./eigen-eigen-5a0156e40feb/test/product_extra.o \
./eigen-eigen-5a0156e40feb/test/product_large.o \
./eigen-eigen-5a0156e40feb/test/product_mmtr.o \
./eigen-eigen-5a0156e40feb/test/product_notemporary.o \
./eigen-eigen-5a0156e40feb/test/product_selfadjoint.o \
./eigen-eigen-5a0156e40feb/test/product_small.o \
./eigen-eigen-5a0156e40feb/test/product_symm.o \
./eigen-eigen-5a0156e40feb/test/product_syrk.o \
./eigen-eigen-5a0156e40feb/test/product_trmm.o \
./eigen-eigen-5a0156e40feb/test/product_trmv.o \
./eigen-eigen-5a0156e40feb/test/product_trsolve.o \
./eigen-eigen-5a0156e40feb/test/qr.o \
./eigen-eigen-5a0156e40feb/test/qr_colpivoting.o \
./eigen-eigen-5a0156e40feb/test/qr_fullpivoting.o \
./eigen-eigen-5a0156e40feb/test/qtvector.o \
./eigen-eigen-5a0156e40feb/test/rand.o \
./eigen-eigen-5a0156e40feb/test/real_qz.o \
./eigen-eigen-5a0156e40feb/test/redux.o \
./eigen-eigen-5a0156e40feb/test/ref.o \
./eigen-eigen-5a0156e40feb/test/resize.o \
./eigen-eigen-5a0156e40feb/test/rvalue_types.o \
./eigen-eigen-5a0156e40feb/test/schur_complex.o \
./eigen-eigen-5a0156e40feb/test/schur_real.o \
./eigen-eigen-5a0156e40feb/test/selfadjoint.o \
./eigen-eigen-5a0156e40feb/test/simplicial_cholesky.o \
./eigen-eigen-5a0156e40feb/test/sizeof.o \
./eigen-eigen-5a0156e40feb/test/sizeoverflow.o \
./eigen-eigen-5a0156e40feb/test/smallvectors.o \
./eigen-eigen-5a0156e40feb/test/sparseLM.o \
./eigen-eigen-5a0156e40feb/test/sparse_basic.o \
./eigen-eigen-5a0156e40feb/test/sparse_block.o \
./eigen-eigen-5a0156e40feb/test/sparse_permutations.o \
./eigen-eigen-5a0156e40feb/test/sparse_product.o \
./eigen-eigen-5a0156e40feb/test/sparse_ref.o \
./eigen-eigen-5a0156e40feb/test/sparse_solvers.o \
./eigen-eigen-5a0156e40feb/test/sparse_vector.o \
./eigen-eigen-5a0156e40feb/test/sparselu.o \
./eigen-eigen-5a0156e40feb/test/sparseqr.o \
./eigen-eigen-5a0156e40feb/test/special_numbers.o \
./eigen-eigen-5a0156e40feb/test/spqr_support.o \
./eigen-eigen-5a0156e40feb/test/stable_norm.o \
./eigen-eigen-5a0156e40feb/test/stddeque.o \
./eigen-eigen-5a0156e40feb/test/stddeque_overload.o \
./eigen-eigen-5a0156e40feb/test/stdlist.o \
./eigen-eigen-5a0156e40feb/test/stdlist_overload.o \
./eigen-eigen-5a0156e40feb/test/stdvector.o \
./eigen-eigen-5a0156e40feb/test/stdvector_overload.o \
./eigen-eigen-5a0156e40feb/test/superlu_support.o \
./eigen-eigen-5a0156e40feb/test/swap.o \
./eigen-eigen-5a0156e40feb/test/triangular.o \
./eigen-eigen-5a0156e40feb/test/umeyama.o \
./eigen-eigen-5a0156e40feb/test/umfpack_support.o \
./eigen-eigen-5a0156e40feb/test/unalignedassert.o \
./eigen-eigen-5a0156e40feb/test/unalignedcount.o \
./eigen-eigen-5a0156e40feb/test/upperbidiagonalization.o \
./eigen-eigen-5a0156e40feb/test/vectorization_logic.o \
./eigen-eigen-5a0156e40feb/test/vectorwiseop.o \
./eigen-eigen-5a0156e40feb/test/visitor.o \
./eigen-eigen-5a0156e40feb/test/zerosized.o 

CU_DEPS += \
./eigen-eigen-5a0156e40feb/test/cuda_basic.d 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/test/adjoint.d \
./eigen-eigen-5a0156e40feb/test/array.d \
./eigen-eigen-5a0156e40feb/test/array_for_matrix.d \
./eigen-eigen-5a0156e40feb/test/array_of_string.d \
./eigen-eigen-5a0156e40feb/test/array_replicate.d \
./eigen-eigen-5a0156e40feb/test/array_reverse.d \
./eigen-eigen-5a0156e40feb/test/bandmatrix.d \
./eigen-eigen-5a0156e40feb/test/basicstuff.d \
./eigen-eigen-5a0156e40feb/test/bdcsvd.d \
./eigen-eigen-5a0156e40feb/test/bicgstab.d \
./eigen-eigen-5a0156e40feb/test/block.d \
./eigen-eigen-5a0156e40feb/test/boostmultiprec.d \
./eigen-eigen-5a0156e40feb/test/bug1213.d \
./eigen-eigen-5a0156e40feb/test/bug1213_main.d \
./eigen-eigen-5a0156e40feb/test/cholesky.d \
./eigen-eigen-5a0156e40feb/test/cholmod_support.d \
./eigen-eigen-5a0156e40feb/test/commainitializer.d \
./eigen-eigen-5a0156e40feb/test/conjugate_gradient.d \
./eigen-eigen-5a0156e40feb/test/conservative_resize.d \
./eigen-eigen-5a0156e40feb/test/constructor.d \
./eigen-eigen-5a0156e40feb/test/corners.d \
./eigen-eigen-5a0156e40feb/test/ctorleak.d \
./eigen-eigen-5a0156e40feb/test/denseLM.d \
./eigen-eigen-5a0156e40feb/test/dense_storage.d \
./eigen-eigen-5a0156e40feb/test/determinant.d \
./eigen-eigen-5a0156e40feb/test/diagonal.d \
./eigen-eigen-5a0156e40feb/test/diagonalmatrices.d \
./eigen-eigen-5a0156e40feb/test/dontalign.d \
./eigen-eigen-5a0156e40feb/test/dynalloc.d \
./eigen-eigen-5a0156e40feb/test/eigen2support.d \
./eigen-eigen-5a0156e40feb/test/eigensolver_complex.d \
./eigen-eigen-5a0156e40feb/test/eigensolver_generalized_real.d \
./eigen-eigen-5a0156e40feb/test/eigensolver_generic.d \
./eigen-eigen-5a0156e40feb/test/eigensolver_selfadjoint.d \
./eigen-eigen-5a0156e40feb/test/evaluators.d \
./eigen-eigen-5a0156e40feb/test/exceptions.d \
./eigen-eigen-5a0156e40feb/test/fastmath.d \
./eigen-eigen-5a0156e40feb/test/first_aligned.d \
./eigen-eigen-5a0156e40feb/test/geo_alignedbox.d \
./eigen-eigen-5a0156e40feb/test/geo_eulerangles.d \
./eigen-eigen-5a0156e40feb/test/geo_homogeneous.d \
./eigen-eigen-5a0156e40feb/test/geo_hyperplane.d \
./eigen-eigen-5a0156e40feb/test/geo_orthomethods.d \
./eigen-eigen-5a0156e40feb/test/geo_parametrizedline.d \
./eigen-eigen-5a0156e40feb/test/geo_quaternion.d \
./eigen-eigen-5a0156e40feb/test/geo_transformations.d \
./eigen-eigen-5a0156e40feb/test/half_float.d \
./eigen-eigen-5a0156e40feb/test/hessenberg.d \
./eigen-eigen-5a0156e40feb/test/householder.d \
./eigen-eigen-5a0156e40feb/test/incomplete_cholesky.d \
./eigen-eigen-5a0156e40feb/test/inplace_decomposition.d \
./eigen-eigen-5a0156e40feb/test/integer_types.d \
./eigen-eigen-5a0156e40feb/test/inverse.d \
./eigen-eigen-5a0156e40feb/test/is_same_dense.d \
./eigen-eigen-5a0156e40feb/test/jacobi.d \
./eigen-eigen-5a0156e40feb/test/jacobisvd.d \
./eigen-eigen-5a0156e40feb/test/linearstructure.d \
./eigen-eigen-5a0156e40feb/test/lscg.d \
./eigen-eigen-5a0156e40feb/test/lu.d \
./eigen-eigen-5a0156e40feb/test/mapped_matrix.d \
./eigen-eigen-5a0156e40feb/test/mapstaticmethods.d \
./eigen-eigen-5a0156e40feb/test/mapstride.d \
./eigen-eigen-5a0156e40feb/test/meta.d \
./eigen-eigen-5a0156e40feb/test/metis_support.d \
./eigen-eigen-5a0156e40feb/test/miscmatrices.d \
./eigen-eigen-5a0156e40feb/test/mixingtypes.d \
./eigen-eigen-5a0156e40feb/test/mpl2only.d \
./eigen-eigen-5a0156e40feb/test/nesting_ops.d \
./eigen-eigen-5a0156e40feb/test/nomalloc.d \
./eigen-eigen-5a0156e40feb/test/nullary.d \
./eigen-eigen-5a0156e40feb/test/numext.d \
./eigen-eigen-5a0156e40feb/test/packetmath.d \
./eigen-eigen-5a0156e40feb/test/pardiso_support.d \
./eigen-eigen-5a0156e40feb/test/pastix_support.d \
./eigen-eigen-5a0156e40feb/test/permutationmatrices.d \
./eigen-eigen-5a0156e40feb/test/prec_inverse_4x4.d \
./eigen-eigen-5a0156e40feb/test/product_extra.d \
./eigen-eigen-5a0156e40feb/test/product_large.d \
./eigen-eigen-5a0156e40feb/test/product_mmtr.d \
./eigen-eigen-5a0156e40feb/test/product_notemporary.d \
./eigen-eigen-5a0156e40feb/test/product_selfadjoint.d \
./eigen-eigen-5a0156e40feb/test/product_small.d \
./eigen-eigen-5a0156e40feb/test/product_symm.d \
./eigen-eigen-5a0156e40feb/test/product_syrk.d \
./eigen-eigen-5a0156e40feb/test/product_trmm.d \
./eigen-eigen-5a0156e40feb/test/product_trmv.d \
./eigen-eigen-5a0156e40feb/test/product_trsolve.d \
./eigen-eigen-5a0156e40feb/test/qr.d \
./eigen-eigen-5a0156e40feb/test/qr_colpivoting.d \
./eigen-eigen-5a0156e40feb/test/qr_fullpivoting.d \
./eigen-eigen-5a0156e40feb/test/qtvector.d \
./eigen-eigen-5a0156e40feb/test/rand.d \
./eigen-eigen-5a0156e40feb/test/real_qz.d \
./eigen-eigen-5a0156e40feb/test/redux.d \
./eigen-eigen-5a0156e40feb/test/ref.d \
./eigen-eigen-5a0156e40feb/test/resize.d \
./eigen-eigen-5a0156e40feb/test/rvalue_types.d \
./eigen-eigen-5a0156e40feb/test/schur_complex.d \
./eigen-eigen-5a0156e40feb/test/schur_real.d \
./eigen-eigen-5a0156e40feb/test/selfadjoint.d \
./eigen-eigen-5a0156e40feb/test/simplicial_cholesky.d \
./eigen-eigen-5a0156e40feb/test/sizeof.d \
./eigen-eigen-5a0156e40feb/test/sizeoverflow.d \
./eigen-eigen-5a0156e40feb/test/smallvectors.d \
./eigen-eigen-5a0156e40feb/test/sparseLM.d \
./eigen-eigen-5a0156e40feb/test/sparse_basic.d \
./eigen-eigen-5a0156e40feb/test/sparse_block.d \
./eigen-eigen-5a0156e40feb/test/sparse_permutations.d \
./eigen-eigen-5a0156e40feb/test/sparse_product.d \
./eigen-eigen-5a0156e40feb/test/sparse_ref.d \
./eigen-eigen-5a0156e40feb/test/sparse_solvers.d \
./eigen-eigen-5a0156e40feb/test/sparse_vector.d \
./eigen-eigen-5a0156e40feb/test/sparselu.d \
./eigen-eigen-5a0156e40feb/test/sparseqr.d \
./eigen-eigen-5a0156e40feb/test/special_numbers.d \
./eigen-eigen-5a0156e40feb/test/spqr_support.d \
./eigen-eigen-5a0156e40feb/test/stable_norm.d \
./eigen-eigen-5a0156e40feb/test/stddeque.d \
./eigen-eigen-5a0156e40feb/test/stddeque_overload.d \
./eigen-eigen-5a0156e40feb/test/stdlist.d \
./eigen-eigen-5a0156e40feb/test/stdlist_overload.d \
./eigen-eigen-5a0156e40feb/test/stdvector.d \
./eigen-eigen-5a0156e40feb/test/stdvector_overload.d \
./eigen-eigen-5a0156e40feb/test/superlu_support.d \
./eigen-eigen-5a0156e40feb/test/swap.d \
./eigen-eigen-5a0156e40feb/test/triangular.d \
./eigen-eigen-5a0156e40feb/test/umeyama.d \
./eigen-eigen-5a0156e40feb/test/umfpack_support.d \
./eigen-eigen-5a0156e40feb/test/unalignedassert.d \
./eigen-eigen-5a0156e40feb/test/unalignedcount.d \
./eigen-eigen-5a0156e40feb/test/upperbidiagonalization.d \
./eigen-eigen-5a0156e40feb/test/vectorization_logic.d \
./eigen-eigen-5a0156e40feb/test/vectorwiseop.d \
./eigen-eigen-5a0156e40feb/test/visitor.d \
./eigen-eigen-5a0156e40feb/test/zerosized.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/test/%.o: ../eigen-eigen-5a0156e40feb/test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

eigen-eigen-5a0156e40feb/test/%.o: ../eigen-eigen-5a0156e40feb/test/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

