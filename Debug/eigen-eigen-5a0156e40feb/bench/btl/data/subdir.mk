################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../eigen-eigen-5a0156e40feb/bench/btl/data/mean.cxx \
../eigen-eigen-5a0156e40feb/bench/btl/data/regularize.cxx \
../eigen-eigen-5a0156e40feb/bench/btl/data/smooth.cxx 

CXX_DEPS += \
./eigen-eigen-5a0156e40feb/bench/btl/data/mean.d \
./eigen-eigen-5a0156e40feb/bench/btl/data/regularize.d \
./eigen-eigen-5a0156e40feb/bench/btl/data/smooth.d 

OBJS += \
./eigen-eigen-5a0156e40feb/bench/btl/data/mean.o \
./eigen-eigen-5a0156e40feb/bench/btl/data/regularize.o \
./eigen-eigen-5a0156e40feb/bench/btl/data/smooth.o 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/bench/btl/data/%.o: ../eigen-eigen-5a0156e40feb/bench/btl/data/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/bench/btl/data" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


