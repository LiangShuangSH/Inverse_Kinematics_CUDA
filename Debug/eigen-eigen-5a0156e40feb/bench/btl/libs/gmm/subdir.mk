################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../eigen-eigen-5a0156e40feb/bench/btl/libs/gmm/main.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/bench/btl/libs/gmm/main.o 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/bench/btl/libs/gmm/main.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/bench/btl/libs/gmm/%.o: ../eigen-eigen-5a0156e40feb/bench/btl/libs/gmm/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/bench/btl/libs/gmm" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

