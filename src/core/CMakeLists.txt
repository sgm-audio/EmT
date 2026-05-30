add_library(core STATIC
ssm_block.cpp
ssm_inference_core.cpp
arena_allocator.cpp
weight_loader.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)
