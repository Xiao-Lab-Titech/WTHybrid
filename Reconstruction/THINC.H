#ifndef _THINC_H_
#define _THINC_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"

class THINC : public Reconstruction 
{
	public :
		THINC(Solver* slv);
		~THINC(void);
		void update(std::vector< std::vector<cellBoundary> >& q_bdry, const std::vector< std::vector<double> >& q);
		std::string getName() const {std::string s = "THINC";return s;}


	private :
		double calcTHINC(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;

		double beta_;
		double T1_;
		
};


THINC::THINC(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);

	beta_ = 1.6;
	T1_ = tanh(0.5*beta_);
}

THINC::~THINC(void) {

}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void THINC::update(std::vector< std::vector<cellBoundary> >& q_bdry, const std::vector< std::vector<double> >& q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
        q_bdry[i+1][0].L = calcTHINC(q[i-1][0], q[i][0], q[i+1][0]);
        q_bdry[i][0].R = calcTHINC(q[i+1][0], q[i][0], q[i-1][0]);
        q_bdry[i+1][1].L = calcTHINC(q[i-1][1], q[i][1], q[i+1][1]);
        q_bdry[i][1].R = calcTHINC(q[i+1][1], q[i][1], q[i-1][1]);
        q_bdry[i+1][2].L = calcTHINC(q[i-1][2], q[i][2], q[i+1][2]);
        q_bdry[i][2].R = calcTHINC(q[i+1][2], q[i][2], q[i-1][2]);
	}
}

double THINC::calcTHINC(double qm1, double q0, double qp1) const {
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

#endif 