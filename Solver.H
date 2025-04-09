#ifndef _SOLVER_H_
#define _SOLVER_H_
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
//#include <vector>
#include <math.h>
#include <string>
#include <tuple>
#include <cmath>
#include <algorithm>


struct cellBoundary {
	double L;
	double R;
};

class Reconstruction;
class Flux;
class TimeIntegral;
class BoundaryCondition;
class Exact;


class Solver 
{
	public : 
		Solver(void);
		~Solver(void);

        void makePreProcessedData(const std::string file);
        void simpleSolve(unsigned int idx_problem, std::tuple<bool, std::string, int> preprocessed_data_info, bool is_logging_data);
		void write2File(const std::string file) const;
		void plot(const char* file) const;
		
		int getNcell() const {return N_cell_;}
		int getNmax() const {return N_max_;}
		int getGhostcell() const {return gs_;}
		double getDX() const {return dx_;}
		double getDT() const {return dt_;}
		double getT() const {return t_;}
		double* getX() {return x_;}

		void setReconstruction(Reconstruction* recons) {recons_ = recons;}
		void setFlux(Flux* flux) {fluxer_ = flux;}
		void setTimeIntegral(TimeIntegral* ti) {time_integral_ = ti;}
		void setSelector(int idx_variable, int idx_x, int flag) {selector_[idx_variable][idx_x] = flag;}

		void showInfo() const;

	private :
		void initProblem(unsigned int idx_problem);
		void updateDT();

	private :
		int N_cell_; // The number of cells
		double xl_, xr_; // left/right-side of x value
		int N_max_; // The number of cells including ghost cells
		int gs_;
		double* x_; // vector of x
		double** q_; // numerical solution. conservative variable
		double** qe_; // Exact solution. primitive variable
		double** flux_; // numerical flux
		cellBoundary** q_bdry_; // cell boundary by reconstruction
		Flux* fluxer_;
		BoundaryCondition* bc_;
		Reconstruction* recons_;
		TimeIntegral* time_integral_;
		Exact* exact_;
		double cfl_; // Courant number
		double t_; // simulating time
		double dx_;
		double dt_;
		int ts_; // time step
		double te_; // time at end of simulation
		unsigned int idx_problem_;
		int** selector_; // indicating which candidate function BVD scheme selects.
};

#endif