add_executable(
    apophis
        main.cpp
        apophis_driver.cpp
        hydro/hydro.cpp
        refinement/gradient.cpp
        refinement/other.cpp
)

add_subdirectory(pgen)

target_link_libraries(apophis PRIVATE Parthenon::parthenon)
target_link_libraries(apophis PRIVATE singularity-eos)
