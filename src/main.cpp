// Parthenon headers
#include "parthenon_manager.hpp"
#include "utils/error_checking.hpp"

// Apophis headers
#include "apophis_driver.hpp"
#include "pgen/pgen.hpp"

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
  if (problem == "sod") {
    pman.app_input->ProblemGenerator = sod::ProblemGenerator;
  } else if (problem == "linear_wave") {
    pman.app_input->InitUserMeshData = linear_wave::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop;
  } else if (problem == "blast") {
    pman.app_input->ProblemGenerator = blast::ProblemGenerator;
  } else if (problem == "unset") {
    PARTHENON_FAIL("[Apophis]: Problem unset. Exiting.");
  } else {
    PARTHENON_FAIL("[Apophis]: Problem not recognized. Exiting.");
  }

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "[Apophis]: Initializing..." << std::endl;
  }
  pman.ParthenonInitPackagesAndMesh();

  Apophis::ApophisDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());

  auto driver_status = driver.Execute();

  pman.ParthenonFinalize();

  return EXIT_SUCCESS;
}
