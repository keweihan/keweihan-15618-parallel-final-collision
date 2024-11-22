# demos/CMakeLists.txt

# Define demo executables
set(DEMO_TARGETS
    pong
    collisionStress
    conway
    piCalcMomentum
)

# First copy assets to build directory for Visual Studio development
add_custom_target(copy_assets_to_build ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/demos/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_SOURCE_DIR}/demos/assets 
        ${CMAKE_BINARY_DIR}/demos/assets
    COMMENT "Copying assets to build directory"
)

# Add executables
foreach(target ${DEMO_TARGETS})
    add_executable(${target} ${target}.cpp)
    
    # Link with SimpleECS (this should bring in all dependencies)
    target_link_libraries(${target} PRIVATE SimpleECS)
    
    # Add only necessary include path
    target_include_directories(${target} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
    )
    
    # Copy assets folder to executable directory
    add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/demos/assets
        $<TARGET_FILE_DIR:${target}>/assets
    )
endforeach()