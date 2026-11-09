cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME DM_BMI088)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# ---------------------------------------------------------------------------
# 使用自己维护的链接脚本（DMA 内存布局）。
# 注意：CubeMX 重新“Generate Code”时会重写本文件，这一段在用户块之外，
# 可能被清掉；如果发现 RAM_D1 占用变成 0，就是把这里丢了，重新加回来即可。
# ---------------------------------------------------------------------------
string(REPLACE "STM32H723xG_flash.ld" "STM32H723xG_flash_dma.ld"
       CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
    User/lib
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    User/src/alg_filter.c
    User/src/alg_pid.c
    User/src/BMI088driver.c
    User/src/BMI088Middleware.c
    User/src/bsp_dwt.c
    User/src/ekf.c
    User/src/imu_odom.c
    User/src/remote_control.c
    User/src/app_can.c
    # fivebar.c / lqr.c 已从构建移除：RL 策略不再调用五连杆正解和 LQR
    # （膝盖是被动机械闭链，不需要解算也不需要观测）。
    # 源文件仍留在 User/src 里，需要恢复时把下面两行取消注释即可。
    # User/src/fivebar.c
    # User/src/lqr.c
    User/src/vmc.c
    Task/src/INS_TASK.c
    Task/src/CHASSIS_TASK.c
    Task/src/REMOTE_TASK.c
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
    User/inc
    Task/inc
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx
    # CMSIS-DSP 预编译库（放在 User/lib，CubeMX 重新生成不会清理）
    arm_cortexM7lfdp_math

    # Add user defined libraries
)
