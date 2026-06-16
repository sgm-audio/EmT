set(CLAP_SOURCES
clap_entry.cpp
clap_fx_plugin.cpp
clap_plugin.cpp
clap_params.cpp
clap_note_ports.cpp
clap_state.cpp
clap_audio_ports.cpp
clap_tail.cpp
clap_latency.cpp
clap_presets.cpp
clap_gui.cpp
)

add_library(mamba_synth SHARED ${CLAP_SOURCES})

target_include_directories(mamba_synth
PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/..
${CMAKE_SOURCE_DIR}/extern/clap/include
)

target_link_libraries(mamba_synth PRIVATE core common ara_plugin)

set_target_properties(mamba_synth PROPERTIES
CXX_VISIBILITY_PRESET hidden
VISIBILITY_INLINES_HIDDEN ON
)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
find_package(X11 REQUIRED)
target_link_libraries(mamba_synth PRIVATE ${X11_LIBRARIES})
target_include_directories(mamba_synth PRIVATE ${X11_INCLUDE_DIR})
set_target_properties(mamba_synth PROPERTIES
OUTPUT_NAME "MambaSynth"
SUFFIX ".clap"
PREFIX ""
)
install(TARGETS mamba_synth DESTINATION lib/clap)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
set_target_properties(mamba_synth PROPERTIES
OUTPUT_NAME "MambaSynth"
SUFFIX ""
PREFIX ""
)
install(TARGETS mamba_synth DESTINATION /Library/Audio/Plug-Ins/CLAP)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
set_target_properties(mamba_synth PROPERTIES
OUTPUT_NAME "MambaSynth"
SUFFIX ".clap"
PREFIX ""
)
install(TARGETS mamba_synth DESTINATION "C:/Program Files/Common Files/CLAP")
endif()
