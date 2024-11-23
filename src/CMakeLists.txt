add_executable(
    apophis
        main.cpp
        apophis_driver.cpp
)

target_link_libraries(apophis PRIVATE Parthenon::parthenon)
