add_library(ara_plugin STATIC
ara_document_controller.cpp
ara_playback_renderer.cpp
ara_editor_renderer.cpp
)
target_include_directories(ara_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(ara_plugin PRIVATE core common)
set_target_properties(ara_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ARA_INCLUDE_DIR)
    target_include_directories(ara_plugin PRIVATE ${ARA_INCLUDE_DIR})
endif()
