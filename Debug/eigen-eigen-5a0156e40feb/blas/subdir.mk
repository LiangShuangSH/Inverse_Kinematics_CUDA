################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../eigen-eigen-5a0156e40feb/blas/complex_double.cpp \
../eigen-eigen-5a0156e40feb/blas/complex_single.cpp \
../eigen-eigen-5a0156e40feb/blas/double.cpp \
../eigen-eigen-5a0156e40feb/blas/single.cpp \
../eigen-eigen-5a0156e40feb/blas/xerbla.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/blas/complex_double.o \
./eigen-eigen-5a0156e40feb/blas/complex_single.o \
./eigen-eigen-5a0156e40feb/blas/double.o \
./eigen-eigen-5a0156e40feb/blas/single.o \
./eigen-eigen-5a0156e40feb/blas/xerbla.o 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/blas/complex_double.d \
./eigen-eigen-5a0156e40feb/blas/complex_single.d \
./eigen-eigen-5a0156e40feb/blas/double.d \
./eigen-eigen-5a0156e40feb/blas/single.d \
./eigen-eigen-5a0156e40feb/blas/xerbla.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/blas/%.o: ../eigen-eigen-5a0156e40feb/blas/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/blas" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


