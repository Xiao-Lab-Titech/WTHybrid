#ifndef _HLLCFLUX_H_ 
#define _HLLCFLUX_H_ 
#include "../Flux.H" 
#include "../Solver.H"
#include "../Parameter.H"

class HLLCFlux : public Flux 
{
	public : 
		HLLCFlux(Solver* slv) : solver_(slv) {solver_->setFlux(this);}
		~HLLCFlux(void);
		void update(double** flux, const cellBoundary* const* q_bdry);

		double getMCS() const {return max_characteristic_speed_;}
		std::string getName() const {std::string s = "HLLC";return s;}

	private :
		double sign(double a);

    private :
        double max_characteristic_speed_;
		Solver* solver_;
};


void HLLCFlux::update(double** flux, const cellBoundary* const* q_bdry) {
	double N_max = solver_->getNmax();
	double gs = solver_->getGhostcell();

	double mws_x = 0.0;
	double rho_l,rho_r,rhou_l,rhou_r,rhoE_l,rhoE_r; // conservative variables
	double u_l,u_r,p_l,p_r,c_l,c_r; // primitive variables
	double f_rho_l,f_rho_r,f_rhou_l,f_rhou_r,f_rhoE_l,f_rhoE_r; // flux
	double rho_bar,c_bar,p_star,q_l,q_r,S_l,S_r; // PVRS
	double S_star;
	double S_ratio_l,S_ratio_r,rho_lstar,rho_rstar,rhou_lstar,rhou_rstar,rhoE_lstar,rhoE_rstar;
	double s_minus,s_plus;
	for (int i = gs; i <= N_max - gs; i++) {
		switch(RECONSTRUCTION_VAR) {
			case 0:	// reconstructed by conservative
				// conservative variables
				rho_l = q_bdry[0][i].L; rho_r = q_bdry[0][i].R;
				rhou_l = q_bdry[1][i].L; rhou_r = q_bdry[1][i].R;
				rhoE_l = q_bdry[2][i].L; rhoE_r = q_bdry[2][i].R;
				// primitive variables
				u_l = rhou_l / rho_l; u_r = rhou_r / rho_r;
				p_l = (GAMMA - 1.)*(rhoE_l - .5*rhou_l*u_l); p_r = (GAMMA - 1.)*(rhoE_r - .5*rhou_r*u_r);
				c_l = sqrt(GAMMA*p_l/rho_l); c_r = sqrt(GAMMA*p_r/rho_r);
				break;
			case 1: // reconstructed by primitive
				// primitive variables
				rho_l = q_bdry[0][i].L; rho_r = q_bdry[0][i].R;
				u_l = q_bdry[1][i].L; u_r = q_bdry[1][i].R;
				p_l = q_bdry[2][i].L; p_r = q_bdry[2][i].R;
				c_l = sqrt(GAMMA*p_l/rho_l); c_r = sqrt(GAMMA*p_r/rho_r);
				// conservative variables
				rhou_l = rho_l * u_l; rhou_r = rho_r * u_r;
				rhoE_l = 0.5*rhou_l*u_l + p_l/(GAMMA-1.0); rhoE_r = 0.5*rhou_r*u_r + p_r/(GAMMA-1.0);
				break;
			case 2: // reconstructed by characteristic
				// soon
				break;
			default :
				break;
		}

		// flux
        f_rho_l = rhou_l; f_rho_r = rhou_r;
        f_rhou_l = rhou_l * u_l + p_l; f_rhou_r = rhou_r * u_r + p_r;
        f_rhoE_l = (rhoE_l + p_l) * u_l; f_rhoE_r = (rhoE_r + p_r) * u_r;

		// PVRS (Primitive Variable Riemann Solver)
		rho_bar = 0.5*(rho_l+rho_r);
		c_bar = 0.5*(c_l+c_r);
		p_star = fmax(0.0, 0.5*(p_l+p_r) - 0.5*(u_r-u_l)*rho_bar*c_bar);
		q_l = (p_star <= p_l ? 1.0 : sqrt(1.0 + (GAMMA+1.0)/(2.0*GAMMA)*(p_star/p_l - 1.0)));
		q_r = (p_star <= p_r ? 1.0 : sqrt(1.0 + (GAMMA+1.0)/(2.0*GAMMA)*(p_star/p_r - 1.0)));
		S_l = u_l - c_l*q_l; S_r = u_r + c_r*q_r;

		mws_x = fmax(mws_x, fmax(fabs(S_l), fabs(S_r)));

		// speed of contact discontinuity
		S_star = (p_r-p_l + rho_l*u_l*(S_l-u_l) - rho_r*u_r*(S_r-u_r)) / (rho_l*(S_l-u_l) - rho_r*(S_r-u_r));

		// conservative variables at intermediate region
		S_ratio_l = (S_l - u_l)/(S_l - S_star); S_ratio_r = (S_r - u_r)/(S_r - S_star);
		rho_lstar = S_ratio_l*rho_l; rho_rstar = S_ratio_r*rho_r;
		rhou_lstar = rho_lstar*S_star; rhou_rstar = rho_rstar*S_star;
		rhoE_lstar = rho_lstar*(rhoE_l/rho_l + (S_star - u_l)*(S_star + p_l/rho_l/(S_l - u_l))); rhoE_rstar = rho_rstar*(rhoE_r/rho_r + (S_star - u_r)*(S_star + p_r/rho_r/(S_r - u_r)));

		// calculate flux
		s_minus = fmin(S_l,0.0);
		s_plus = fmax(S_r,0.0);
		flux[0][i] = 0.5*(1.0+sign(S_star))*(f_rho_l + s_minus*(rho_lstar - rho_l))
				+ 0.5*(1.0-sign(S_star))*(f_rho_r + s_plus*(rho_rstar - rho_r));
		flux[1][i] = 0.5*(1.0+sign(S_star))*(f_rhou_l + s_minus*(rhou_lstar - rhou_l))
			 	+ 0.5*(1.0-sign(S_star))*(f_rhou_r + s_plus*(rhou_rstar - rhou_r));
		flux[2][i] = 0.5*(1.0+sign(S_star))*(f_rhoE_l + s_minus*(rhoE_lstar - rhoE_l))
			 	+ 0.5*(1.0-sign(S_star))*(f_rhoE_r + s_plus*(rhoE_rstar - rhoE_r));
	}
	max_characteristic_speed_ = mws_x;
}

double HLLCFlux::sign(double a) {
	if (a > 0.0) return 1.0;
	else if (a < 0.0) return -1.0;
	else return 0.0;
}

#endif
