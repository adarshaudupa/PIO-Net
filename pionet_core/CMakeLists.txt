cmake_minimum_required(VERSION 3.13)

# ------------------------------------------------------------------------------
# Portable Raspberry Pi Pico SDK Import (Works on Linux, macOS, and Windows)
# Reads PICO_SDK_PATH from environment variable or cmake cache
# ------------------------------------------------------------------------------
if (DEFINED ENV{PICO_SDK_PATH} AND (NOT PICO_SDK_PATH))
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
    message("Using PICO_SDK_PATH from environment: '${PICO_SDK_PATH}'")
endif ()

if (DEFINED PICO_SDK_PATH)
    if (NOT EXISTS "${PICO_SDK_PATH}/pico_sdk_init.cmake")
        message(FATAL_ERROR "Directory specified by PICO_SDK_PATH '${PICO_SDK_PATH}' does not contain pico_sdk_init.cmake")
    endif ()
    set(PICO_SDK_PATH "${PICO_SDK_PATH}" CACHE PATH "Path to the Raspberry Pi Pico SDK")
    include("${PICO_SDK_PATH}/pico_sdk_init.cmake")
elseif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/pico_sdk_import.cmake")
    include(${CMAKE_CURRENT_LIST_DIR}/pico_sdk_import.cmake)
else ()
    message(WARNING "PICO_SDK_PATH is not set. Run: export PICO_SDK_PATH=/path/to/pico-sdk")
endif ()

project(pionet_core C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize Raspberry Pi Pico SDK
if (COMMAND pico_sdk_init)
    pico_sdk_init()
    add_compile_options(-O3)
endif ()

add_executable(pionet_core
        model_data.h
        stage2_infer.h
        stage2_infer.c
        stage2_features.h
        stage2_features.c)

# Generate C header from PIO assembly
pico_generate_pio_header(pionet_core ${CMAKE_CURRENT_LIST_DIR}/pdm_reader.pio)
pico_generate_pio_header(pionet_core ${CMAKE_CURRENT_LIST_DIR}/pdm_stage1.pio)


target_sources(pionet_core PRIVATE main.c)

target_include_directories(pionet_core PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(pionet_core PRIVATE
    pico_stdlib
    hardware_pio
    hardware_gpio
        hardware_dma
        pico_multicore
)

option(PIONET_ENABLE_STDIO_USB "Enable USB stdio output" ON)
option(PIONET_VARIANT2_POWER "Build variant2 with USB stdio, telemetry, and status LED disabled for power measurements" OFF)

if (PIONET_VARIANT2_POWER)
    set(PIONET_ENABLE_STDIO_USB OFF)
    target_compile_definitions(pionet_core PRIVATE PIONET_TELEMETRY=0 PIONET_STATUS_LED=0)
endif ()

pico_add_extra_outputs(pionet_core)
if (PIONET_ENABLE_STDIO_USB)
    pico_enable_stdio_usb(pionet_core 1)
else ()
    pico_enable_stdio_usb(pionet_core 0)
endif ()
pico_enable_stdio_uart(pionet_core 0)