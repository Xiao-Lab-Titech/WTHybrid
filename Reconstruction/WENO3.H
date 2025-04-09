#ifndef _WENO3_H_
#define _WENO3_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"

class WENO3 : public Reconstruction 
{
	public :
		WENO3(Solver* slv);
		~WENO3(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "WENO3";return s;}

	private :
		void transform(int name_of_var, const double* const* q);
		double calcWENO3(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;
		std::vector<std::vector<double>> q_tmp_;
		
};


WENO3::WENO3(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
	int N_max = solver_->getNmax();
	q_tmp_.assign(N_max, std::vector<double>(3, 0.0));

}

WENO3::~WENO3(void) {
	std::vector<std::vector<double>>().swap(q_tmp_);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void WENO3::update(cellBoundary** q_bdry, const double* const* q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	transform(RECONSTRUCTION_VAR, q);

	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			q_bdry[i][j+1].L = calcWENO3(q_tmp_[i][j-1], q_tmp_[i][j], q_tmp_[i][j+1]);
			q_bdry[i][j].R = calcWENO3(q_tmp_[i][j+1], q_tmp_[i][j], q_tmp_[i][j-1]);
		}
	}
}

void WENO3::transform(int name_of_var, const double* const* q) {
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

double WENO3::calcWENO3(double qm1, double q0, double qp1) const {
	double qL; // L value is to i+1/2
	double ul0, ul1, w0, w1, alp0, alp1, beta0, beta1;
	double global_smooth_indicator, eps;
	double ideal_weight0 = 1.0/3.0, ideal_weight1 = 2.0/3.0;
	ul0 = (3.0*q0 - qm1)*0.5;
	ul1 = (q0 + qp1)*0.5;

	beta0 = (q0-qm1)*(q0-qm1);
	beta1 = (q0-qp1)*(q0-qp1);

	global_smooth_indicator = fabs(beta0-beta1);
	eps = 1.0e-15;
	alp0 = ideal_weight0*(1.0 + global_smooth_indicator/(beta0+eps));
	alp1 = ideal_weight1*(1.0 + global_smooth_indicator/(beta1+eps));

	w0 = alp0/(alp0+alp1);
	w1 = alp1/(alp0+alp1);

	qL =  w0*ul0 + w1*ul1;
	return qL;
}

#endif 