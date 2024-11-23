add_executable(
    apophis
        main.cpp
        apophis_driver.cpp
)

add_subdirectory(pgen)

target_link_libraries(apophis PRIVATE Parthenon::parthenon)
