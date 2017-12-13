################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/BVH_Example.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/EulerAngles.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/FFT.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixExponential.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixFunction.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixLogarithm.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixPower.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixPower_optimal.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSine.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSinh.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSquareRoot.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/PolynomialSolver1.cpp \
../eigen-eigen-5a0156e40feb/unsupported/doc/examples/PolynomialUtils1.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/BVH_Example.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/EulerAngles.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/FFT.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixExponential.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixFunction.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixLogarithm.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixPower.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixPower_optimal.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSine.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSinh.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSquareRoot.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/PolynomialSolver1.o \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/PolynomialUtils1.o 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/BVH_Example.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/EulerAngles.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/FFT.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixExponential.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixFunction.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixLogarithm.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixPower.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixPower_optimal.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSine.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSinh.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/MatrixSquareRoot.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/PolynomialSolver1.d \
./eigen-eigen-5a0156e40feb/unsupported/doc/examples/PolynomialUtils1.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/unsupported/doc/examples/%.o: ../eigen-eigen-5a0156e40feb/unsupported/doc/examples/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/unsupported/doc/examples" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


