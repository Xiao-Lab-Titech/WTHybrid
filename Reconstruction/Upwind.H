#ifndef _Upwind_H_
#define _Upwind_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"

class Upwind : public Reconstruction 
{
	public :
		Upwind(Solver* slv);
		~Upwind(void);
		void update(std::vector< std::vector<cellBoundary> >& q_bdry, const std::vector< std::vector<double> >& q);
		std::string getName() const {std::string s = "Upwind";return s;}

	private :
		Solver* solver_;
		
};


Upwind::Upwind(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
}

Upwind::~Upwind(void) {

}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void Upwind::update(std::vector< std::vector<cellBoundary> >& q_bdry, const std::vector< std::vector<double> >& q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
        q_bdry[i+1][0].L = q[i][0];
        q_bdry[i][0].R = q[i][0];
        q_bdry[i+1][1].L = q[i][1];
        q_bdry[i][1].R = q[i][1];
        q_bdry[i+1][2].L = q[i][2];
        q_bdry[i][2].R = q[i][2];
	}
}



#endif 