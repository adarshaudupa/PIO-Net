# Standard Raspberry Pi Pico SDK import script
# Pulls in the SDK from PICO_SDK_PATH if defined, otherwise provides guidance.

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
else ()
    message(FATAL_ERROR
            "SDK location was not specified. Please set the PICO_SDK_PATH environment variable:\n"
            "  export PICO_SDK_PATH=/path/to/pico-sdk\n"
            "or pass -DPICO_SDK_PATH=/path/to/pico-sdk to cmake.")
endif ()
