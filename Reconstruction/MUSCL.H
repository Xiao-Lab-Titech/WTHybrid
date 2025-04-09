#ifndef _MUSCL_H_
#define _MUSCL_H_ 
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"
#include "../RawVector.H"

class MUSCL : public Reconstruction 
{
	public :
		MUSCL(Solver* slv);
		~MUSCL(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "MUSCL";return s;}
		double getReconstructionTime() const {return recon_time_;}

	private :
		void transform(int name_of_var, const double* const* q);
		double calcMUSCL(double qm1, double q0, double qp1) const;
		double minmod(double r) const;
		double superbee(double r) const;
		double vanLeer(double r) const;
		double vanAlbada(double r) const;
		double sign(double x) const;

	private :
		Solver* solver_;

		double recon_time_;
	
		double** q_tmp_;
};


MUSCL::MUSCL(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
	recon_time_ = 0.0;

	int N_max = solver_->getNmax();
	initVector2d<double>(q_tmp_, 3, N_max, 0.0);
}

MUSCL::~MUSCL(void) {
	//int N_max = solver_->getNmax();
	freeVector2d<double>(q_tmp_, 3);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void MUSCL::update(cellBoundary** q_bdry, const double* const* q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	double start = omp_get_wtime();
	transform(RECONSTRUCTION_VAR, q);

	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			q_bdry[i][j+1].L = calcMUSCL(q_tmp_[i][j-1], q_tmp_[i][j], q_tmp_[i][j+1]);
			q_bdry[i][j].R = calcMUSCL(q_tmp_[i][j+1], q_tmp_[i][j], q_tmp_[i][j-1]);
		}
	}
	double end = omp_get_wtime();
	//end = std::chrono::system_clock::now(); // 計測終了
	//double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	double elapsed = end - start;
	//printf("recon time: %f\n", elapsed);
	recon_time_ += elapsed;
}

void MUSCL::transform(int name_of_var, const double* const* q) {
	int N_max = solver_->getNmax();
	switch(name_of_var) {
		case 0:	// reconstructed by conservative
			for (int i = 0; i < N_max; i++) {
				q_tmp_[0][i] = q[0][i];
				q_tmp_[1][i] = q[1][i];
				q_tmp_[2][i] = q[2][i];
			}
			break;
		case 1: // reconstructed by primitive
			for (int i = 0; i < N_max; i++) {
				q_tmp_[0][i] = q[0][i];
				q_tmp_[1][i] = q[1][i] / q_tmp_[0][i];
				q_tmp_[2][i] = (GAMMA - 1.0)*(q[2][i] - 0.5*q_tmp_[0][i]*q_tmp_[1][i]*q_tmp_[1][i]);
			}
			break;
		case 2: // reconstructed by characteristic
			// soon
			break;
		default :
			break;
	}
}

double MUSCL::calcMUSCL(double qm1, double q0, double qp1) const {
	double qL;
	double phi_L, r_L;
	
	r_L = (q0 - qm1)/(qp1 - q0 + sign(qp1 - q0)*1.0e-5);
	phi_L = vanLeer(r_L);
	qL = q0 + 0.5*phi_L*(qp1 - q0);

	return qL;
}

double MUSCL::minmod(double r) const {
	return std::max(0.0, std::min(1.0, r));
}

double MUSCL::superbee(double r) const {
	return std::max(0.0, std::max(std::min(1.0, 2.0*r), std::min(2.0, r)));
}

double MUSCL::vanLeer(double r) const {
	double abs_r = std::abs(r);
	return (r + abs_r)/(1.0 + abs_r);
}

double MUSCL::vanAlbada(double r) const {
	double rr = r*r;
	return (r + rr)/(1.0 + rr);
}

double MUSCL::sign(double x) const {
	if(x >= 0.0) return 1.0;
	else return -1.0;
}

#endif 