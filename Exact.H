#ifndef _EXACT_H_
#define _EXACT_H_
#include "Solver.H"
#include "Parameter.H"
// Exact solution of 1D Riemann problem for general initial values

// Reference: E.F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009).

struct Variable {
	double L;
	double R;
};


class Exact {
	public :
        Exact(Solver* slv) : solver_(slv) {}
		~Exact(){}

		void update(double** qe);
        void setProblem(double rho_l, double u_l, double p_l, double rho_r, double u_r, double p_r, double xc_);
        void initialize();

    private :
        double fK(double rhoK, double pK, double cK, double p) const;
        double dfK(double rhoK, double pK, double cK, double p) const;

    private :
        Solver* solver_;
        Variable rho_, u_, p_;
        Variable c_;
        Variable rho_star_, S_, SH_, ST_;
        double xc_;
        double p_star_, u_star_;

};

void Exact::update(double** qe) {
    int gs = solver_->getGhostcell();
    int N_cell = solver_->getNcell();
    double t = solver_->getT();
    double* x = solver_->getX();

    for (int i = 0; i < N_cell; i++) {
        double S=(x[i]-xc_)/t;
        if (S < u_star_) { // Left
            if (p_star_ > p_.L) { // shock
                if (S < S_.L) {
                    qe[0][i+gs] = rho_.L;
                    qe[1][i+gs] = u_.L;
                    qe[2][i+gs] = p_.L;
                } else {
                    qe[0][i+gs] = rho_star_.L;
                    qe[1][i+gs] = u_star_;
                    qe[2][i+gs] = p_star_;
                }
            } else { // rarefaction
                if (S < SH_.L) {
                    qe[0][i+gs] = rho_.L;
                    qe[1][i+gs] = u_.L;
                    qe[2][i+gs] = p_.L;
                } else if (S > ST_.L) {
                    qe[0][i+gs] = rho_star_.L;
                    qe[1][i+gs] = u_star_;
                    qe[2][i+gs] = p_star_;
                } else {
                    double u_fan_L = 2.0/(GAMMA+1.0)*(S+c_.L+(GAMMA-1.0)/2.0*u_.L);
                    double p_fan_L = p_.L * pow(2.0/(GAMMA+1.0)+(GAMMA-1.0)/(GAMMA+1.0)*(u_.L-S)/c_.L, (2.0*GAMMA)/(GAMMA-1.0));
                    double rho_fan_L=pow(p_fan_L/p_.L, 1.0/GAMMA) * rho_.L;
                    qe[0][i+gs] = rho_fan_L;
                    qe[1][i+gs] = u_fan_L;
                    qe[2][i+gs] = p_fan_L;
                }
            }

        } else { // Right
            if (p_star_ > p_.R) { // shock
                if (S > S_.R) {
                    qe[0][i+gs] = rho_.R;
                    qe[1][i+gs] = u_.R;
                    qe[2][i+gs] = p_.R;
                } else {
                    qe[0][i+gs] = rho_star_.R;
                    qe[1][i+gs] = u_star_;
                    qe[2][i+gs] = p_star_;
                }
            } else { // rarefaction
                if (S > SH_.R) {
                    qe[0][i+gs] = rho_.R;
                    qe[1][i+gs] = u_.R;
                    qe[2][i+gs] = p_.R;
                } else if (S < ST_.R) {
                    qe[0][i+gs] = rho_star_.R;
                    qe[1][i+gs] = u_star_;
                    qe[2][i+gs] = p_star_;
                } else {
                    double u_fan_R = 2.0/(GAMMA+1.0)*(S-c_.R+(GAMMA-1.0)/2.0*u_.R);
                    double p_fan_R = p_.R * pow(2.0/(GAMMA+1.0)-(GAMMA-1.0)/(GAMMA+1.0)*(u_.R-S)/c_.R, (2.0*GAMMA)/(GAMMA-1.0));
                    double rho_fan_R = pow(p_fan_R/p_.R, 1.0/GAMMA) * rho_.R;
                    qe[0][i+gs] = rho_fan_R;
                    qe[1][i+gs] = u_fan_R;
                    qe[2][i+gs] = p_fan_R;
                }
            }
        }
    }


}

void Exact::setProblem(double rho_l, double u_l, double p_l, double rho_r, double u_r, double p_r, double xc) {
    rho_.L = rho_l, rho_.R = rho_r;
    u_.L = u_l, u_.R = u_r;
    p_.L = p_l, p_.R = p_r;
    xc_ = xc;
}

void Exact::initialize() {
    c_ = {sqrt(GAMMA*p_.L/rho_.L), sqrt(GAMMA*p_.R/rho_.R)};
    double TOL = 1.0e-6;

    // Estimate initial value of p*
    double p0 = std::max(TOL,(p_.L+p_.R)/2+(u_.L-u_.R)*(rho_.L+rho_.R)*(c_.L+c_.R)/8); // PVRS
    double p_itr = p0, p_new;
    int count = 0;


    try {
        while(1) { // Newton's iteration method
            count = count + 1;
            double f = fK(rho_.L,p_.L,c_.L,p_itr) + fK(rho_.R,p_.R,c_.R,p_itr) + u_.R - u_.L; 
            double df = dfK(rho_.L,p_.L,c_.L,p_itr) + dfK(rho_.R,p_.R,c_.R,p_itr);
            p_new = p_itr - f/df;
            if (p_new <= 0) {
                double alf = 1.0;
                while(1) {
                    alf = alf/2.0;
                    p_new = p_itr - alf*f/df;
                    if (p_new > 0) break;
                    
                    if (alf < 2.0e-20) {
                        throw std::runtime_error("nagative p*. See initialize() in Exact.H.");
                    }
                }   
            }
            if (abs(p_new - p_itr)/((p_new + p_itr)/2.0) < TOL) break;
            p_itr = p_new;
            if (count > 1000) {
                throw std::runtime_error("p* does not converge. See initialize() in Exact.H.");
            }
        }
    } catch (std::runtime_error e) {
        std::cerr << "Simulation error : " << e.what() << std::endl;
        abort();
    }

    p_star_ = p_new;

    u_star_ = u_.L - fK(rho_.L,p_.L,c_.L,p_star_);

    // Find rhoL*, rhoR* and wave speeds
    if (p_star_ > p_.L) { // Left shock
        rho_star_.L = ((GAMMA-1.0)/(GAMMA+1.0)+p_star_/p_.L)/(1.0+(GAMMA-1.0)/(GAMMA+1.0)*p_star_/p_.L)*rho_.L;
        S_.L = u_.L - c_.L*sqrt(1.0+(GAMMA+1.0)/(2.0*GAMMA)*(p_star_/p_.L-1.0));
    } else { // Left rarefaction
        rho_star_.L = pow(p_star_/p_.L, 1.0/GAMMA) * rho_.L;
        SH_.L = u_.L - c_.L;
        ST_.L = u_star_ - sqrt(GAMMA*p_star_/rho_star_.L);
    }
    
    if (p_star_ > p_.R) { // Right shock
        rho_star_.R = ((GAMMA-1.0)/(GAMMA+1.0)+p_star_/p_.R)/(1.0+(GAMMA-1.0)/(GAMMA+1.0)*p_star_/p_.R)*rho_.R;
        S_.R = u_.R + c_.R*sqrt(1.0+(GAMMA+1.0)/(2.0*GAMMA)*(p_star_/p_.R-1.0));
    } else { // Right rarefaction
        rho_star_.R = pow(p_star_/p_.R, 1.0/GAMMA) * rho_.R;
        SH_.R = u_.R + c_.R;
        ST_.R = u_star_ + sqrt(GAMMA*p_star_/rho_star_.R);
    }
    
    //printf("count = %d, rho_star_L = %f, rho_star_R = %f, p_star = %f, u_star = %f\n", count, rho_star_.L, rho_star_.R, p_star_, u_star_);
    //printf("S_L = %f, SH_L = %f, ST_L = %f\n", S_.L, SH_.L, ST_.L);
    //printf("S_R = %f, SH_R = %f, ST_R = %f\n", S_.R, SH_.R, ST_.R);
}

// velocity difference function
inline double Exact::fK(double rhoK, double pK, double cK, double p) const {
    if (p > pK) {// shock
        return (p - pK)/sqrt(rhoK/2.0*((GAMMA+1.0)*p+(GAMMA-1.0)*pK));
    } else { // rarefaction
        return 2.0*cK/(GAMMA-1.0)*(pow(p/pK, (GAMMA-1.0)/(2.0*GAMMA)) - 1.0);
    }
}

// derivative of velocity difference function
inline double Exact::dfK(double rhoK, double pK, double cK, double p) const {
    if (p > pK) {// shock
        double AK = 2.0/((GAMMA+1.0)*rhoK);
        double BK = (GAMMA-1)/(GAMMA+1)*pK;
        return sqrt(AK/(p+BK))*(1.0-(p-pK)/(2*(p+BK)));
    } else { // rarefaction
        return pow(p/pK, -(GAMMA+1.0)/(2.0*GAMMA))/(rhoK*cK);
    }
}

#endif
