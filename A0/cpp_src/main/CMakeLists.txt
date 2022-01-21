set (MAINHEAD
    )

set ( MAINSRC
    main.cpp
)
SET( executable   Assignment0X )
add_executable(${executable} ${MAINSRC} ${MAINHEAD})
INCLUDE_DIRECTORIES   ("${PROJECT_SOURCE_DIR}/solver")

target_link_libraries(${executable} PUBLIC solver Eigen3::Eigen)


