#ifndef _BOUNDARYCONDITION_H_
#define _BOUNDARYCONDITION_H_
#include "Solver.H"

//class Solver;

enum bdry {open, periodic};

class BoundaryCondition {
	public :
        BoundaryCondition(Solver* slv) : solver_(slv) {}
		~BoundaryCondition(){}

		void update(double** q);
        void setBC(int l, int r) {bc_l_ = l; bc_r_ = r;}
        void getInfo();

    private :
        Solver* solver_;
        int bc_l_;
        int bc_r_;
};

void BoundaryCondition::update(double** q) {
    int gs = solver_->getGhostcell();
    int N_max = solver_->getNmax();
	switch (bc_l_) {
		case open:
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < gs; j++){
                    q[i][j] = q[i][gs];
                }
            }  
			break;
		case periodic:
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < gs; j++){
                    q[i][j] = q[i][N_max - 2*gs + j - 1];
                }
            }   
			break;
		default :
			//debagging
			break;
	}

	switch (bc_r_) {
		case open:
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < gs; j++){
                    q[i][N_max - 1 - j] = q[i][N_max - gs - 1];
                }
            }  
			break;
		case periodic:
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < gs; j++){
                    q[i][N_max - 1 - j] = q[i][2*gs - j];
                }
            }   
			break;
		default :
			//debagging
			break;
	}
}


#endif
