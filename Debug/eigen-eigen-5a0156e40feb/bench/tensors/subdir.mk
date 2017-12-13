################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_fp16_gpu.cu \
../eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_gpu.cu 

CC_SRCS += \
../eigen-eigen-5a0156e40feb/bench/tensors/benchmark_main.cc \
../eigen-eigen-5a0156e40feb/bench/tensors/contraction_benchmarks_cpu.cc \
../eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_cpu.cc \
../eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_sycl.cc 

CC_DEPS += \
./eigen-eigen-5a0156e40feb/bench/tensors/benchmark_main.d \
./eigen-eigen-5a0156e40feb/bench/tensors/contraction_benchmarks_cpu.d \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_cpu.d \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_sycl.d 

OBJS += \
./eigen-eigen-5a0156e40feb/bench/tensors/benchmark_main.o \
./eigen-eigen-5a0156e40feb/bench/tensors/contraction_benchmarks_cpu.o \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_cpu.o \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_fp16_gpu.o \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_gpu.o \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_sycl.o 

CU_DEPS += \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_fp16_gpu.d \
./eigen-eigen-5a0156e40feb/bench/tensors/tensor_benchmarks_gpu.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/bench/tensors/%.o: ../eigen-eigen-5a0156e40feb/bench/tensors/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/bench/tensors" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

eigen-eigen-5a0156e40feb/bench/tensors/%.o: ../eigen-eigen-5a0156e40feb/bench/tensors/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/bench/tensors" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


