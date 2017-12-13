################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_argmax_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_cast_float16_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_complex_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_contract_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_device.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_float16_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_random_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction_cuda.cu \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_scan_cuda.cu 

CPP_SRCS += \
../eigen-eigen-5a0156e40feb/unsupported/test/BVH.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/EulerAngles.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/FFT.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/FFTW.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/NonLinearOptimization.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/NumericalDiff.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/alignedvector3.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/autodiff.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/autodiff_scalar.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_eventcount.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_meta.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_non_blocking_thread_pool.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_runqueue.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_argmax.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_assign.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_broadcast_sycl.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_broadcasting.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_casts.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_chipping.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_comparisons.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_concatenation.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_const.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_contraction.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_convolution.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_custom_index.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_custom_op.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_device_sycl.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_dimension.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_empty.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_expr.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_fft.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_fixed_size.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_forced_eval.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_generator.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_ifft.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_image_patch.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_index_list.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_inflation.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_intdiv.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_io.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_layout_swap.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_lvalue.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_map.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_math.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_mixed_indices.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_morphing.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_notification.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_complex.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_const_values.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_strings.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_padding.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_patch.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_random.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction_sycl.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_ref.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reverse.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_roundings.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_scan.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_shuffling.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_simple.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_striding.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_sugar.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_sycl.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_symmetry.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_thread_pool.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_uint128.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_volume_patch.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/dgmres.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/forward_adolc.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/gmres.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/kronecker_product.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/levenberg_marquardt.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/matrix_exponential.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/matrix_function.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/matrix_power.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/matrix_square_root.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/minres.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/mpreal_support.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/openglsupport.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/polynomialsolver.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/polynomialutils.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/sparse_extra.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/special_functions.cpp \
../eigen-eigen-5a0156e40feb/unsupported/test/splines.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/unsupported/test/BVH.o \
./eigen-eigen-5a0156e40feb/unsupported/test/EulerAngles.o \
./eigen-eigen-5a0156e40feb/unsupported/test/FFT.o \
./eigen-eigen-5a0156e40feb/unsupported/test/FFTW.o \
./eigen-eigen-5a0156e40feb/unsupported/test/NonLinearOptimization.o \
./eigen-eigen-5a0156e40feb/unsupported/test/NumericalDiff.o \
./eigen-eigen-5a0156e40feb/unsupported/test/alignedvector3.o \
./eigen-eigen-5a0156e40feb/unsupported/test/autodiff.o \
./eigen-eigen-5a0156e40feb/unsupported/test/autodiff_scalar.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_eventcount.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_meta.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_non_blocking_thread_pool.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_runqueue.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_argmax.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_argmax_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_assign.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_broadcast_sycl.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_broadcasting.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_cast_float16_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_casts.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_chipping.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_comparisons.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_complex_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_concatenation.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_const.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_contract_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_contraction.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_convolution.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_custom_index.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_custom_op.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_device.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_device_sycl.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_dimension.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_empty.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_expr.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_fft.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_fixed_size.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_forced_eval.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_forced_eval_sycl.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_generator.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_ifft.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_image_patch.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_index_list.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_inflation.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_intdiv.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_io.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_layout_swap.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_lvalue.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_map.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_math.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_mixed_indices.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_morphing.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_notification.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_complex.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_const_values.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_float16_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_strings.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_padding.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_patch.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_random.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_random_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction_sycl.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_ref.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reverse.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_roundings.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_scan.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_scan_cuda.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_shuffling.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_simple.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_striding.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_sugar.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_sycl.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_symmetry.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_thread_pool.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_uint128.o \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_volume_patch.o \
./eigen-eigen-5a0156e40feb/unsupported/test/dgmres.o \
./eigen-eigen-5a0156e40feb/unsupported/test/forward_adolc.o \
./eigen-eigen-5a0156e40feb/unsupported/test/gmres.o \
./eigen-eigen-5a0156e40feb/unsupported/test/kronecker_product.o \
./eigen-eigen-5a0156e40feb/unsupported/test/levenberg_marquardt.o \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_exponential.o \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_function.o \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_power.o \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_square_root.o \
./eigen-eigen-5a0156e40feb/unsupported/test/minres.o \
./eigen-eigen-5a0156e40feb/unsupported/test/mpreal_support.o \
./eigen-eigen-5a0156e40feb/unsupported/test/openglsupport.o \
./eigen-eigen-5a0156e40feb/unsupported/test/polynomialsolver.o \
./eigen-eigen-5a0156e40feb/unsupported/test/polynomialutils.o \
./eigen-eigen-5a0156e40feb/unsupported/test/sparse_extra.o \
./eigen-eigen-5a0156e40feb/unsupported/test/special_functions.o \
./eigen-eigen-5a0156e40feb/unsupported/test/splines.o 

CU_DEPS += \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_argmax_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_cast_float16_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_complex_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_contract_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_device.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_float16_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_random_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction_cuda.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_scan_cuda.d 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/unsupported/test/BVH.d \
./eigen-eigen-5a0156e40feb/unsupported/test/EulerAngles.d \
./eigen-eigen-5a0156e40feb/unsupported/test/FFT.d \
./eigen-eigen-5a0156e40feb/unsupported/test/FFTW.d \
./eigen-eigen-5a0156e40feb/unsupported/test/NonLinearOptimization.d \
./eigen-eigen-5a0156e40feb/unsupported/test/NumericalDiff.d \
./eigen-eigen-5a0156e40feb/unsupported/test/alignedvector3.d \
./eigen-eigen-5a0156e40feb/unsupported/test/autodiff.d \
./eigen-eigen-5a0156e40feb/unsupported/test/autodiff_scalar.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_eventcount.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_meta.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_non_blocking_thread_pool.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_runqueue.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_argmax.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_assign.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_broadcast_sycl.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_broadcasting.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_casts.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_chipping.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_comparisons.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_concatenation.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_const.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_contraction.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_convolution.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_custom_index.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_custom_op.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_device_sycl.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_dimension.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_empty.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_expr.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_fft.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_fixed_size.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_forced_eval.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_forced_eval_sycl.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_generator.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_ifft.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_image_patch.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_index_list.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_inflation.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_intdiv.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_io.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_layout_swap.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_lvalue.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_map.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_math.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_mixed_indices.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_morphing.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_notification.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_complex.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_const_values.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_of_strings.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_padding.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_patch.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_random.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reduction_sycl.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_ref.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_reverse.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_roundings.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_scan.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_shuffling.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_simple.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_striding.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_sugar.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_sycl.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_symmetry.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_thread_pool.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_uint128.d \
./eigen-eigen-5a0156e40feb/unsupported/test/cxx11_tensor_volume_patch.d \
./eigen-eigen-5a0156e40feb/unsupported/test/dgmres.d \
./eigen-eigen-5a0156e40feb/unsupported/test/forward_adolc.d \
./eigen-eigen-5a0156e40feb/unsupported/test/gmres.d \
./eigen-eigen-5a0156e40feb/unsupported/test/kronecker_product.d \
./eigen-eigen-5a0156e40feb/unsupported/test/levenberg_marquardt.d \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_exponential.d \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_function.d \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_power.d \
./eigen-eigen-5a0156e40feb/unsupported/test/matrix_square_root.d \
./eigen-eigen-5a0156e40feb/unsupported/test/minres.d \
./eigen-eigen-5a0156e40feb/unsupported/test/mpreal_support.d \
./eigen-eigen-5a0156e40feb/unsupported/test/openglsupport.d \
./eigen-eigen-5a0156e40feb/unsupported/test/polynomialsolver.d \
./eigen-eigen-5a0156e40feb/unsupported/test/polynomialutils.d \
./eigen-eigen-5a0156e40feb/unsupported/test/sparse_extra.d \
./eigen-eigen-5a0156e40feb/unsupported/test/special_functions.d \
./eigen-eigen-5a0156e40feb/unsupported/test/splines.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/unsupported/test/%.o: ../eigen-eigen-5a0156e40feb/unsupported/test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/unsupported/test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

eigen-eigen-5a0156e40feb/unsupported/test/%.o: ../eigen-eigen-5a0156e40feb/unsupported/test/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/unsupported/test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


