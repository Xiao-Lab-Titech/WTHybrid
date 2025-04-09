#include "Solver.H"
#include "./Reconstruction/W3TBVD.H"
#include "./Reconstruction/MTBVD.H"
//#include "./Reconstruction/WENO3.H"
//#include "./Reconstruction/THINC.H"
//#include "./Reconstruction/Upwind.H"
//#include "./Reconstruction/Polynominal.H"
#include "./Reconstruction/MUSCL.H"
#include "./Reconstruction/MLBasedW3TBVD.H"
#include "./Reconstruction/MLBasedMTBVD.H"
#include "./Flux/HLLCFlux.H"
#include "./TimeIntegral/RK3.H"
#include <chrono>


int main (int argc, char *argv[]) 
{
	const char* onnx_path = "./ONNX/Huangnet.onnx";
	bool isGPU = false;

	Solver *solver = new Solver();
	
	new HLLCFlux(solver);

	//W3TBVD *recon = new W3TBVD(solver);
	//MTBVD *recon = new MTBVD(solver);
	//new WENO3(solver);
	//new THINC(solver);
	//new Upwind(solver);
	//new Polynominal(solver);
	//new MUSCL(solver);
	//MLBasedW3TBVD *recon = new MLBasedW3TBVD(solver, onnx_path, isGPU);
	MLBasedMTBVD *recon = new MLBasedMTBVD(solver, onnx_path, isGPU);
	
	new RK3(solver);
	
	solver->simpleSolve(0, {false, "./PreProcessedData/", 50}, false);
	//printf("transform: %f, reconstruction: %f, TBV: %f, selection: %f\n", recon->getTransformTime(), recon->getreconstructionTime(), recon->getTBVTime(), recon->getSelectionTime());
	printf("transform: %f, inference: %f, selection: %f\n", recon->getTransformTime(), recon->getInferenceTime(), recon->getSelectionTime());
	// Solve(solver1, solver2, Exact);
	return 0;  
} 
 
