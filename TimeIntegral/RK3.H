#ifndef _RK3_H_
#define _RK3_H_
#include "../TimeIntegral.H"
#include "../Solver.H"
#include "../RawVector.H"

class RK3 : public TimeIntegral
{
	public :
		RK3(Solver* slv);
		~RK3(void);
		void update(double** q, const double* const* flux);
		std::string getName() const {std::string s = "RK3";return s;}

		int getSubstep() const {return substep_;}
		int getMaxSubstep() const {return max_substep_;}
		void setSubstep(int s) {substep_ = s;}
        
    private :
		Solver* solver_;
		int substep_;
		const int max_substep_ = 3;
	
		double** q_star_;
		double** q_tmp_;
};

inline RK3::RK3(Solver* slv) : solver_(slv) {
	solver_->setTimeIntegral(this);

	substep_ = 1;
	int N_max = solver_->getNmax();
	initVector2d<double>(q_star_, 3, N_max, 0.0);
	initVector2d<double>(q_tmp_, 3, N_max, 0.0);
}

inline RK3::~RK3() {
	//int N_max = solver_->getNmax();
	freeVector2d<double>(q_star_, 3);
	freeVector2d<double>(q_tmp_, 3);
}

void RK3::update(double** q, const double* const* flux) {
	double N_max = solver_->getNmax();
	double gs = solver_->getGhostcell();
	double dx = solver_->getDX();
	double dt = solver_->getDT();

	switch (substep_) {
		case 1:
			for (int i = 0; i < 3; i++) {
				for (int j = gs; j < N_max - gs; j++) {
					q_star_[i][j] = q[i][j] + dt * (flux[i][j]-flux[i][j+1]) / dx;
					q_tmp_[i][j] = q[i][j];
					q[i][j] = q_star_[i][j];

				}
			}
			break;
		case 2:
			for (int i = 0; i < 3; i++) {
				for (int j = gs; j < N_max - gs; j++) {
					q[i][j] = 0.75*q_tmp_[i][j] + 0.25*q_star_[i][j] + 0.25 * dt * (flux[i][j]-flux[i][j+1]) / dx;
					q_star_[i][j] = q[i][j]; // q**
				}
			}
			break;
		case 3:
			for (int i = 0; i < 3; i++) {
				for (int j = gs; j < N_max - gs; j++) {
					q[i][j] = 1.0/3.0*q_tmp_[i][j] + 2.0/3.0*q_star_[i][j] + 2.0/3.0 * dt * (flux[i][j]-flux[i][j+1]) / dx;
				}
			}
			break;
		default :
			//debagging
			break;
	}

}


#endif
