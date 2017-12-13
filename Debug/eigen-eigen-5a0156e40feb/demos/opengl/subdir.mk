################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../eigen-eigen-5a0156e40feb/demos/opengl/camera.cpp \
../eigen-eigen-5a0156e40feb/demos/opengl/gpuhelper.cpp \
../eigen-eigen-5a0156e40feb/demos/opengl/icosphere.cpp \
../eigen-eigen-5a0156e40feb/demos/opengl/quaternion_demo.cpp \
../eigen-eigen-5a0156e40feb/demos/opengl/trackball.cpp 

OBJS += \
./eigen-eigen-5a0156e40feb/demos/opengl/camera.o \
./eigen-eigen-5a0156e40feb/demos/opengl/gpuhelper.o \
./eigen-eigen-5a0156e40feb/demos/opengl/icosphere.o \
./eigen-eigen-5a0156e40feb/demos/opengl/quaternion_demo.o \
./eigen-eigen-5a0156e40feb/demos/opengl/trackball.o 

CPP_DEPS += \
./eigen-eigen-5a0156e40feb/demos/opengl/camera.d \
./eigen-eigen-5a0156e40feb/demos/opengl/gpuhelper.d \
./eigen-eigen-5a0156e40feb/demos/opengl/icosphere.d \
./eigen-eigen-5a0156e40feb/demos/opengl/quaternion_demo.d \
./eigen-eigen-5a0156e40feb/demos/opengl/trackball.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/demos/opengl/%.o: ../eigen-eigen-5a0156e40feb/demos/opengl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/demos/opengl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


