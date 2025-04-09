#ifndef _RECONSTRUCTION_H_
#define _RECONSTRUCTION_H_

class Solver;

class Reconstruction 
{
	public :
		Reconstruction(void) {};
		Reconstruction(Solver* slv) : solver_(slv) {solver_->setReconstruction(this);}
		~Reconstruction(void) {};
		virtual void update(cellBoundary** q_bdry, const double* const* q) = 0;
		virtual std::string getName() const = 0;
		virtual double getReconstructionTime() const = 0;


	private :
		virtual void transform(int name_of_var, const double* const* q) = 0;

	private :
		Solver* solver_;

};


#endif
