SET ( SRC
    solver.cpp
     )

SET ( HEADERS
    solver.h
    )


add_library ( solver  ${SRC} ${HEADERS} )
target_link_libraries (solver PUBLIC Eigen3::Eigen)
target_compile_features(solver INTERFACE cxx_std_17)

