SET ( SRC
    RandomWalk.cpp
     )

SET ( HEADERS
    RandomWalk.h
    )


add_library ( solver  ${SRC} ${HEADERS} )
target_link_libraries (solver PUBLIC Eigen3::Eigen)
target_compile_features(solver INTERFACE cxx_std_17)

