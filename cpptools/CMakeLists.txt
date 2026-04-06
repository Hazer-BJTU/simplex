cmake_minimum_required(VERSION 3.20)

project(cpptools LANGUAGES CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORT OUTPUT LTO_CHECK_OUTPUT)

message(STATUS "LTO support: ${LTO_SUPPORT}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_RELEASE OFF)
option(BUILD_UNITEST OFF)
option(QUIET_MODE OFF)

set(Boost_USE_STATIC_LIBS ON)

find_package(Boost REQUIRED COMPONENTS "filesystem" "system" "program_options")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/libs")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/third_party/include")
link_directories(${PROJECT_SOURCE_DIR}/third_party/libs)

set(THIRD_PARTY_LIB_NAMES "tree-sitter" "tree-sitter-python")
set(THIRD_PARTY_LIBS "")

foreach(LIB_NAME ${THIRD_PARTY_LIB_NAMES})
    find_library(${LIB_NAME}_TARGET NAMES ${LIB_NAME} PATHS "${PROJECT_SOURCE_DIR}/third_party/libs")
    if(NOT ${LIB_NAME}_TARGET)
        message(STATUS "[Error] - third party library ${LIB_NAME} not found")
    else()
        message(STATUS "[Library] - third party library found ${LIB_NAME}: ${${LIB_NAME}_TARGET}")
    endif()
    list(APPEND THIRD_PARTY_LIBS ${${LIB_NAME}_TARGET})
endforeach()

message(STATUS "[Options]- build_release: ${BUILD_RELEASE}, build_unitest: ${BUILD_UNITEST}, server_quiet_mode: ${QUIET_MODE}")

file(GLOB SOURCES "src/*.cpp")
file(GLOB LANG_SOURCES "src/languages/*.cpp")
file(GLOB TEST_SOURCES "unitest/*.cpp")

if(BUILD_RELEASE)
    set(COMPLIER_OPTION_FLAGS "-O3")
else()
    set(COMPLIER_OPTION_FLAGS "-g3" "-O0")
endif()

message(STATUS "[Options]- compile options: ${COMPLIER_OPTION_FLAGS}")

set(BASE_DEPENDENCIES "Boost::filesystem" "Boost::system")
set(LANG_DEPENDENCIES ${BASE_DEPENDENCIES} ${THIRD_PARTY_LIBS})
set(MAIN_DEPENDENCIES ${BASE_DEPENDENCIES} "Boost::program_options")

set(LANG_LIBRARY "_simplex_lang")
add_library(${LANG_LIBRARY} STATIC ${LANG_SOURCES})
target_compile_options(${LANG_LIBRARY} PUBLIC ${COMPLIER_OPTION_FLAGS})
target_link_libraries(${LANG_LIBRARY} PUBLIC ${LANG_DEPENDENCIES})
if(LTO_SUPPORT AND BUILD_RELEASE)
    set_property(TARGET ${LANG_LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "[Options]- lib${LANG_LIBRARY} enables interprocedural optimization")
endif()

set(EXECUTABLE_NAME "simplex_tool_server")
add_executable(${EXECUTABLE_NAME} ${SOURCES})
target_compile_options(${EXECUTABLE_NAME} PRIVATE ${COMPLIER_OPTION_FLAGS})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${MAIN_DEPENDENCIES} ${LANG_LIBRARY})
if(QUIET_MODE)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE "QUIET_MODE")
    message(STATUS "[Options]- ${EXECUTABLE_NAME} enables quiet mode")
endif()
if(LTO_SUPPORT AND BUILD_RELEASE)
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "[Options]- ${EXECUTABLE_NAME} enables interprocedural optimization")
endif()

install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION bin)

if(BUILD_UNITEST)
    set(SOURCES_WM ${SOURCES})
    list(REMOVE_ITEM SOURCES_WM "${PROJECT_SOURCE_DIR}/src/main.cpp")
    foreach(TEST_PATH ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_PATH} NAME_WE)
        add_executable(${TEST_NAME} ${SOURCES_WM} ${TEST_PATH})
        target_compile_options(${TEST_NAME} PRIVATE ${COMPLIER_OPTION_FLAGS})
        target_link_libraries(${TEST_NAME} PRIVATE ${MAIN_DEPENDENCIES} ${LANG_LIBRARY})
        set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests")
        message(STATUS "[Unitest]- unitest '${TEST_NAME}' will be built")
    endforeach()
endif()
