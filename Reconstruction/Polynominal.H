#ifndef _Polynominal_H_
#define _Polynominal_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"

class Polynominal : public Reconstruction 
{
	public :
		Polynominal(Solver* slv);
		~Polynominal(void);
		void update(std::vector< std::vector<cellBoundary> >& q_bdry, const std::vector< std::vector<double> >& q);
		std::string getName() const {std::string s = "Polynominal";return s;}

	private :
		double calcPoly3(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;
		
};


Polynominal::Polynominal(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
}

Polynominal::~Polynominal(void) {

}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void Polynominal::update(std::vector< std::vector<cellBoundary> >& q_bdry, const std::vector< std::vector<double> >& q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
        q_bdry[i+1][0].L = calcPoly3(q[i-1][0], q[i][0], q[i+1][0]);
        q_bdry[i][0].R = calcPoly3(q[i+1][0], q[i][0], q[i-1][0]);
        q_bdry[i+1][1].L = calcPoly3(q[i-1][1], q[i][1], q[i+1][1]);
        q_bdry[i][1].R = calcPoly3(q[i+1][1], q[i][1], q[i-1][1]);
        q_bdry[i+1][2].L = calcPoly3(q[i-1][2], q[i][2], q[i+1][2]);
        q_bdry[i][2].R = calcPoly3(q[i+1][2], q[i][2], q[i-1][2]);
	}
}

double Polynominal::calcPoly3(double qm1, double q0, double qp1) const {
	double qL; // L value is to i+1/2
	double ul0, ul1;
	double ideal_weight0 = 1.0/3.0, ideal_weight1 = 2.0/3.0;
	ul0 = (3.0*q0 - qm1)*0.5;
	ul1 = (q0 + qp1)*0.5;
	qL =  ideal_weight0*ul0 + ideal_weight1*ul1;
	return qL;
}

#endif 