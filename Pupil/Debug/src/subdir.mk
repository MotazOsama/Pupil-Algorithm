################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CannyEdgeDetector.cpp \
../src/InitialRegionGenerator.cpp \
../src/Pupil.cpp \
../src/PupilDetector.cpp 

OBJS += \
./src/CannyEdgeDetector.o \
./src/InitialRegionGenerator.o \
./src/Pupil.o \
./src/PupilDetector.o 

CPP_DEPS += \
./src/CannyEdgeDetector.d \
./src/InitialRegionGenerator.d \
./src/Pupil.d \
./src/PupilDetector.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


