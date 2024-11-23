#include "main.hpp"

// Parthenon headers
#include "parthenon_manager.hpp"

// Apophis headers
#include "apophis_driver.hpp"

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

  pman.app_input->ProcessPackages = Apophis::ProcessPackages;
  const auto problem =
      pman.pinput->GetOrAddString("job", "problem_id", "unset");

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "[Apophis]: Initializing..." << std::endl;
  }

  Apophis::ApophisDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());

  auto driver_status = driver.Execute();

  pman.ParthenonFinalize();

  return EXIT_SUCCESS;
}
