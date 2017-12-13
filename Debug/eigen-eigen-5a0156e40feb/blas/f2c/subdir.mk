################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../eigen-eigen-5a0156e40feb/blas/f2c/chbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/chpmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/complexdots.c \
../eigen-eigen-5a0156e40feb/blas/f2c/ctbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/d_cnjg.c \
../eigen-eigen-5a0156e40feb/blas/f2c/drotm.c \
../eigen-eigen-5a0156e40feb/blas/f2c/drotmg.c \
../eigen-eigen-5a0156e40feb/blas/f2c/dsbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/dspmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/dtbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/lsame.c \
../eigen-eigen-5a0156e40feb/blas/f2c/r_cnjg.c \
../eigen-eigen-5a0156e40feb/blas/f2c/srotm.c \
../eigen-eigen-5a0156e40feb/blas/f2c/srotmg.c \
../eigen-eigen-5a0156e40feb/blas/f2c/ssbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/sspmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/stbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/zhbmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/zhpmv.c \
../eigen-eigen-5a0156e40feb/blas/f2c/ztbmv.c 

OBJS += \
./eigen-eigen-5a0156e40feb/blas/f2c/chbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/chpmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/complexdots.o \
./eigen-eigen-5a0156e40feb/blas/f2c/ctbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/d_cnjg.o \
./eigen-eigen-5a0156e40feb/blas/f2c/drotm.o \
./eigen-eigen-5a0156e40feb/blas/f2c/drotmg.o \
./eigen-eigen-5a0156e40feb/blas/f2c/dsbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/dspmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/dtbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/lsame.o \
./eigen-eigen-5a0156e40feb/blas/f2c/r_cnjg.o \
./eigen-eigen-5a0156e40feb/blas/f2c/srotm.o \
./eigen-eigen-5a0156e40feb/blas/f2c/srotmg.o \
./eigen-eigen-5a0156e40feb/blas/f2c/ssbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/sspmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/stbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/zhbmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/zhpmv.o \
./eigen-eigen-5a0156e40feb/blas/f2c/ztbmv.o 

C_DEPS += \
./eigen-eigen-5a0156e40feb/blas/f2c/chbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/chpmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/complexdots.d \
./eigen-eigen-5a0156e40feb/blas/f2c/ctbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/d_cnjg.d \
./eigen-eigen-5a0156e40feb/blas/f2c/drotm.d \
./eigen-eigen-5a0156e40feb/blas/f2c/drotmg.d \
./eigen-eigen-5a0156e40feb/blas/f2c/dsbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/dspmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/dtbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/lsame.d \
./eigen-eigen-5a0156e40feb/blas/f2c/r_cnjg.d \
./eigen-eigen-5a0156e40feb/blas/f2c/srotm.d \
./eigen-eigen-5a0156e40feb/blas/f2c/srotmg.d \
./eigen-eigen-5a0156e40feb/blas/f2c/ssbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/sspmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/stbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/zhbmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/zhpmv.d \
./eigen-eigen-5a0156e40feb/blas/f2c/ztbmv.d 


# Each subdirectory must supply rules for building sources it contributes
eigen-eigen-5a0156e40feb/blas/f2c/%.o: ../eigen-eigen-5a0156e40feb/blas/f2c/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 -gencode arch=compute_50,code=sm_50 -m64 -odir "eigen-eigen-5a0156e40feb/blas/f2c" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/opt/ros/indigo/include -I/home/liang/cuda-workspace/eigen-eigen-5a0156e40feb -G -g -O0 --compile -m64  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


