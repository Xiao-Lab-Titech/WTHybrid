#ifndef _MTBVD_H_
#define _MTBVD_H_ 
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"
#include "../RawVector.H"

class MTBVD : public Reconstruction 
{
	public :
		MTBVD(Solver* slv);
		~MTBVD(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "MTBVD";return s;}
		double getReconstructionTime() const {return recon_time_;}
		double getTransformTime() const {return transfrom_time_;}
		double getreconstructionTime() const {return reconstruction_time_;}
		double getTBVTime() const {return TBV_time_;}
		double getSelectionTime() const {return selection_time_;}

	private :
		void transform(int name_of_var, const double* const* q);
		double calcTHINC(double qm1, double q0, double qp1) const;
		double calcMUSCL(double qm1, double q0, double qp1) const;
	
	private : 
		double minmod(double r) const;
		double superbee(double r) const;
		double vanLeer(double r) const;
		double vanAlbada(double r) const;
		double sign(double x) const;

	private :
		Solver* solver_;

		double recon_time_;
		double reconstruction_time_;
		double transfrom_time_;
		double TBV_time_;
		double selection_time_;
	

		double beta_;
		double T1_;
		cellBoundary** q_bdry_tmp_;
		double*** TBV_;
		double** q_tmp_;
};


MTBVD::MTBVD(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
	recon_time_ = 0.0;
	reconstruction_time_ = 0.0;
	transfrom_time_ = 0.0;
	TBV_time_ = 0.0;
	selection_time_ = 0.0;

	beta_ = 1.6;
	T1_ = tanh(0.5*beta_);

	int N_max = solver_->getNmax();
	initVector2d<cellBoundary>(q_bdry_tmp_, 3, N_max, {0.0, 0.0});
	initVector3d<double>(TBV_, 2, 3, N_max, 0.0);
	initVector2d<double>(q_tmp_, 3, N_max, 0.0);
}

MTBVD::~MTBVD(void) {
	//int N_max = solver_->getNmax();
	freeVector2d<cellBoundary>(q_bdry_tmp_, 3);
	freeVector3d<double>(TBV_, 2, 3);
	freeVector2d<double>(q_tmp_, 3);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void MTBVD::update(cellBoundary** q_bdry, const double* const* q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();
	//std::chrono::system_clock::time_point  start, end;
	//start = std::chrono::system_clock::now(); // 計測開始
	//double*** test;
	

	double start = omp_get_wtime();
	//initVector3d<double>(test, N_max, 2, 3, 0.0);
	transform(RECONSTRUCTION_VAR, q);
	double time_stamp1 = omp_get_wtime();
	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			q_bdry[i][j+1].L = calcMUSCL(q_tmp_[i][j-1], q_tmp_[i][j], q_tmp_[i][j+1]);
			q_bdry[i][j].R = calcMUSCL(q_tmp_[i][j+1], q_tmp_[i][j], q_tmp_[i][j-1]);

			q_bdry_tmp_[i][j+1].L = calcTHINC(q_tmp_[i][j-1], q_tmp_[i][j], q_tmp_[i][j+1]);
			q_bdry_tmp_[i][j].R = calcTHINC(q_tmp_[i][j+1], q_tmp_[i][j], q_tmp_[i][j-1]);
		}
	}
	double time_stamp2 = omp_get_wtime();
	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			TBV_[0][i][j] = fabs(q_bdry[i][j].L - q_bdry[i][j].R) + fabs(q_bdry[i][j+1].L - q_bdry[i][j+1].R);
			TBV_[1][i][j] = fabs(q_bdry_tmp_[i][j].L - q_bdry_tmp_[i][j].R) + fabs(q_bdry_tmp_[i][j+1].L - q_bdry_tmp_[i][j+1].R);
		}
	}
	double time_stamp3 = omp_get_wtime();
	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			if (TBV_[0][i][j] > TBV_[1][i][j]) {
				q_bdry[i][j+1].L = q_bdry_tmp_[i][j+1].L;
				q_bdry[i][j].R = q_bdry_tmp_[i][j].R;
				solver_->setSelector(i, j, 1);
			} else {
				solver_->setSelector(i, j, 0);
			}
		}
	}
	//freeVector3d<double>(test, N_max, 2);

	double end = omp_get_wtime();
	//end = std::chrono::system_clock::now(); // 計測終了
	//double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	double elapsed = end - start;
	//printf("recon time: %f\n", elapsed);
	transfrom_time_ += time_stamp1 - start;
	reconstruction_time_ += time_stamp2 - time_stamp1;
	TBV_time_ += time_stamp3 - time_stamp2;
	selection_time_ += end - time_stamp3;
	recon_time_ += elapsed;
	
	

}

void MTBVD::transform(int name_of_var, const double* const* q) {
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

double MTBVD::calcTHINC(double qm1, double q0, double qp1) const {
	double qL; // L value is to i+1/2
	double alpha,epsilon;
	double T2;
	epsilon = 1e-15;
	alpha = 0.0;

    // left-side cell boundary value
	if ((q0-qm1)*(qp1-q0) > epsilon){
		alpha = -(qp1-2.0*q0+qm1)/(qp1-qm1);
		T2 = tanh(0.5*beta_*alpha);
		qL = 0.5*(qp1+qm1)+0.5*(qp1-qm1)*(T1_+T2/T1_)/(1.0+T2);
	} else {
		qL = q0;
	}

	return qL;
}


double MTBVD::calcMUSCL(double qm1, double q0, double qp1) const {
	double qL;
	double phi_L, r_L;
	
	r_L = (q0 - qm1)/(qp1 - q0 + sign(qp1 - q0)*1.0e-5);
	phi_L = vanLeer(r_L);
	qL = q0 + 0.5*phi_L*(qp1 - q0);

	return qL;
}

double MTBVD::minmod(double r) const {
	return std::max(0.0, std::min(1.0, r));
}

double MTBVD::superbee(double r) const {
	return std::max(0.0, std::max(std::min(1.0, 2.0*r), std::min(2.0, r)));
}

double MTBVD::vanLeer(double r) const {
	double abs_r = std::abs(r);
	return (r + abs_r)/(1.0 + abs_r);
}

double MTBVD::vanAlbada(double r) const {
	double rr = r*r;
	return (r + rr)/(1.0 + rr);
}

double MTBVD::sign(double x) const {
	if(x >= 0.0) return 1.0;
	else return -1.0;
}

#endif 