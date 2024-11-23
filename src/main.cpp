// Parthenon headers
#include "parthenon_manager.hpp"

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return EXIT_SUCCESS;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return EXIT_FAILURE;
  }
}
