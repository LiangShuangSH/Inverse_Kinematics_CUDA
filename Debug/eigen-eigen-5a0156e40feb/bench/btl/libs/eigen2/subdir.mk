################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/btl_tiny_eigen2.cpp \
../eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_adv.cpp \
../eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_linear.cpp \
../eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_matmat.cpp \
../eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_vecmat.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/btl_tiny_eigen2.o \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_adv.o \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_linear.o \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_matmat.o \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_vecmat.o 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/btl_tiny_eigen2.d \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_adv.d \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_linear.d \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_matmat.d \
./eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/main_vecmat.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/%.o: ../eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/bench/btl/libs/eigen2" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


