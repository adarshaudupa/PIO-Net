cmake_minimum_required(VERSION 3.13)

# ------------------------------------------------------------------------------
# Portable Raspberry Pi Pico SDK Import
# Auto-heal truncated/unquoted PICO_SDK_PATH if passed with spaces from CLion
# ------------------------------------------------------------------------------
if (DEFINED PICO_SDK_PATH AND NOT EXISTS "${PICO_SDK_PATH}/pico_sdk_init.cmake")
    if (EXISTS "D:/Apps/Raspberry Pi Pico SDK/pico-sdk/pico_sdk_init.cmake")
        set(PICO_SDK_PATH "D:/Apps/Raspberry Pi Pico SDK/pico-sdk" CACHE PATH "Path to the Raspberry Pi Pico SDK" FORCE)
    elseif (EXISTS "D:/pico/sdk/pico_sdk_init.cmake")
        set(PICO_SDK_PATH "D:/pico/sdk" CACHE PATH "Path to the Raspberry Pi Pico SDK" FORCE)
    endif ()
endif ()

if (DEFINED ENV{PICO_SDK_PATH} AND (NOT PICO_SDK_PATH))
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
endif ()

# Auto-heal toolchain paths if not provided by IDE
if (NOT DEFINED ENV{PICO_TOOLCHAIN_PATH} AND EXISTS "D:/Apps/Raspberry Pi Pico SDK/gcc-arm-none-eabi/bin")
    set(ENV{PICO_TOOLCHAIN_PATH} "D:/Apps/Raspberry Pi Pico SDK/gcc-arm-none-eabi/bin")
endif ()
if (NOT DEFINED PICO_TOOLCHAIN_PATH AND EXISTS "D:/Apps/Raspberry Pi Pico SDK/gcc-arm-none-eabi/bin")
    set(PICO_TOOLCHAIN_PATH "D:/Apps/Raspberry Pi Pico SDK/gcc-arm-none-eabi/bin" CACHE PATH "Path to toolchain" FORCE)
endif ()
if (NOT DEFINED PICO_PIOASM_PATH AND EXISTS "D:/Apps/Raspberry Pi Pico SDK/pico-sdk-tools/pioasm.exe")
    set(PICO_PIOASM_PATH "D:/Apps/Raspberry Pi Pico SDK/pico-sdk-tools/pioasm.exe" CACHE FILEPATH "Path to pioasm" FORCE)
endif ()

if (DEFINED PICO_SDK_PATH AND EXISTS "${PICO_SDK_PATH}/pico_sdk_init.cmake")
    set(PICO_SDK_PATH "${PICO_SDK_PATH}" CACHE PATH "Path to the Raspberry Pi Pico SDK" FORCE)
    include("${PICO_SDK_PATH}/pico_sdk_init.cmake")
elseif (EXISTS "D:/pico/sdk/pico_sdk_init.cmake")
    set(PICO_SDK_PATH "D:/pico/sdk" CACHE PATH "Path to the Raspberry Pi Pico SDK" FORCE)
    include("${PICO_SDK_PATH}/pico_sdk_init.cmake")
elseif (EXISTS "D:/Apps/Raspberry Pi Pico SDK/pico-sdk/pico_sdk_init.cmake")
    set(PICO_SDK_PATH "D:/Apps/Raspberry Pi Pico SDK/pico-sdk" CACHE PATH "Path to the Raspberry Pi Pico SDK" FORCE)
    include("${PICO_SDK_PATH}/pico_sdk_init.cmake")
elseif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/pico_sdk_import.cmake")
    include(${CMAKE_CURRENT_LIST_DIR}/pico_sdk_import.cmake)
else ()
    message(FATAL_ERROR "Could not find Raspberry Pi Pico SDK. Please check PICO_SDK_PATH.")
endif ()

project(pionet_comparisons C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize Raspberry Pi Pico SDK
if (COMMAND pico_sdk_init)
    pico_sdk_init()
    add_compile_options(-O3)
endif ()

# Common sources for Stage-2 classifier
set(STAGE2_SOURCES
    stage2_infer.h
    stage2_infer.c
    stage2_features.h
    stage2_features.c
    model_data.h
)

# Common link libraries
set(COMMON_LIBS
    pico_stdlib
    hardware_pio
    hardware_gpio
    hardware_dma
    hardware_clocks
    hardware_pll
    hardware_resets
    pico_multicore
)

# ==============================================================================
# TARGET 1: Config A - Continuous Software VAD (CPU 100% Active Polling)
# ==============================================================================
add_executable(config_a_continuous
    config_a_cpu_continuous.c
    ${STAGE2_SOURCES}
)
pico_generate_pio_header(config_a_continuous ${CMAKE_CURRENT_LIST_DIR}/pdm_reader.pio)
target_include_directories(config_a_continuous PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(config_a_continuous PRIVATE PIONET_TELEMETRY=0 PIONET_STATUS_LED=0)
target_link_libraries(config_a_continuous PRIVATE ${COMMON_LIBS})
pico_add_extra_outputs(config_a_continuous)
pico_enable_stdio_usb(config_a_continuous 0)
pico_enable_stdio_uart(config_a_continuous 0)

# ==============================================================================
# TARGET 2: Config B - Duty-Cycled Buffer-Wake (DMA 10 ms Wake + Software VAD)
# ==============================================================================
add_executable(config_b_dutycycle
    config_b_dutycycle.c
    ${STAGE2_SOURCES}
)
pico_generate_pio_header(config_b_dutycycle ${CMAKE_CURRENT_LIST_DIR}/pdm_reader.pio)
target_include_directories(config_b_dutycycle PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(config_b_dutycycle PRIVATE PIONET_TELEMETRY=0 PIONET_STATUS_LED=0)
target_link_libraries(config_b_dutycycle PRIVATE ${COMMON_LIBS})
pico_add_extra_outputs(config_b_dutycycle)
pico_enable_stdio_usb(config_b_dutycycle 0)
pico_enable_stdio_uart(config_b_dutycycle 0)

# ==============================================================================
# TARGET 3: Config C - PIO-Net (Hardware Matched Filter + Two-Gear DVFS)
# ==============================================================================
add_executable(config_c_pionet
    config_c_pionet.c
    ${STAGE2_SOURCES}
)
pico_generate_pio_header(config_c_pionet ${CMAKE_CURRENT_LIST_DIR}/pdm_reader.pio)
pico_generate_pio_header(config_c_pionet ${CMAKE_CURRENT_LIST_DIR}/pdm_stage1.pio)
target_include_directories(config_c_pionet PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(config_c_pionet PRIVATE PIONET_TELEMETRY=0 PIONET_STATUS_LED=0)
target_link_libraries(config_c_pionet PRIVATE -Wl,--wrap=set_sys_clock_pll ${COMMON_LIBS})
pico_add_extra_outputs(config_c_pionet)
pico_enable_stdio_usb(config_c_pionet 0)
pico_enable_stdio_uart(config_c_pionet 0)
