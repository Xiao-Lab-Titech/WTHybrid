#ifndef _TIMEINTEGRAL_H_
#define _TIMEINTEGRAL_H_
//#include "Solver.H"

class Solver;

class TimeIntegral
{
	public :
		TimeIntegral(void){}
		TimeIntegral(Solver* slv) : solver_(slv) {solver_->setTimeIntegral(this);}
		~TimeIntegral(void){}
		virtual void update(double** q, const double* const* flux) = 0;

		virtual int getSubstep() const = 0;
		virtual int getMaxSubstep() const = 0;
		virtual void setSubstep(int s) = 0;
		virtual std::string getName() const = 0;


	private :
		Solver* solver_;

};

#endif
