#include "Solver.H"
#include "Reconstruction.H"
#include "Flux.H"
#include "TimeIntegral.H"
#include "BoundaryCondition.H"
#include "Parameter.H"
#include "Exact.H"
#include "RawVector.H"


Solver::Solver(void) {
	N_cell_ = N_CELL;
	xl_ = XL; xr_ = XR;
    gs_ = GHOST_CELL;
	N_max_ = N_cell_ + 2 * gs_;
	dx_ = fabs(xl_ - xr_)/N_cell_;
	cfl_ = CFL;
	t_ = 0.0;
	dt_ = 0.0;
	ts_ = 0;
	initVector1d<double>(x_, N_cell_, 0.0);
	for(int i = 0; i < N_cell_; i++){
		x_[i] = xl_ + dx_/2.0 + dx_*i;
		//printf("x[%d]=%f\n", i,x_[i]);
	}
	initVector2d<double>(q_, 3, N_max_, 0.0);
	initVector2d<double>(qe_, 3, N_max_, 0.0);
	initVector2d<cellBoundary>(q_bdry_, 3, N_max_, {0.0, 0.0});
	initVector2d<double>(flux_, 3, N_max_, 0.0);
	initVector2d<int>(selector_, 3, N_max_, 0);
}

Solver::~Solver(void) {
	delete bc_;
	delete exact_;
	freeVector1d<double>(x_);
	freeVector2d<double>(q_, 3);
	freeVector2d<double>(qe_, 3);
	freeVector2d<cellBoundary>(q_bdry_, 3);
	freeVector2d<double>(flux_, 3);
	freeVector2d<int>(selector_, 3);
}


void Solver::initProblem(unsigned int idx_problem) {
	idx_problem_ = idx_problem;
	double rho0, u0, p0;
	double rho_L, u_L, p_L, rho_R, u_R, p_R, xc;
	exact_ = new Exact(this);
	bc_ = new BoundaryCondition(this);
	if (idx_problem_ == 0) { // Sod
		rho_L = 1.0; u_L = 0.0; p_L = 1.0;
		rho_R = 0.125; u_R = 0.0; p_R = 0.1;
		xc = 0.5*(xl_+xr_);
	    for(int i = 0; i < N_cell_; i++) {
			if(x_[i] < xc) {
				rho0 = rho_L; u0 = u_L; p0 = p_L;
			} else {
				rho0 = rho_R; u0 = u_R; p0 = p_R;
			}
			qe_[0][i+gs_] = rho0;
			qe_[1][i+gs_] = u0;
			qe_[2][i+gs_] = p0;
			q_[0][i+gs_] = rho0;
			q_[1][i+gs_] = rho0*u0;
			q_[2][i+gs_] = 0.5*q_[1][i+gs_] * u0 + p0 / (GAMMA - 1.0);
		}
		bc_->setBC(open, open);
		te_ = 0.25;
		exact_->setProblem(rho_L, u_L, p_L, rho_R, u_R, p_R, xc);
		exact_->initialize();
	} else if (idx_problem_ == 1) { // Lax
		rho_L = 0.445; u_L = 0.698; p_L = 3.528;
		rho_R = 0.5; u_R = 0.0; p_R = 0.571;
		xc = 0.5*(xl_+xr_);
	    for(int i = 0; i < N_cell_; i++) {
			if(x_[i] < xc) {
				rho0 = rho_L; u0 = u_L; p0 = p_L;
			} else {
				rho0 = rho_R; u0 = u_R; p0 = p_R;
			}
			qe_[0][i+gs_] = rho0;
			qe_[1][i+gs_] = u0;
			qe_[2][i+gs_] = p0;
			q_[0][i+gs_] = rho0;
			q_[1][i+gs_] = rho0*u0;
			q_[2][i+gs_] = 0.5*q_[1][i+gs_] * u0 + p0 / (GAMMA - 1.0);
		}
		bc_->setBC(open, open);
		te_ = 0.16;
		exact_->setProblem(rho_L, u_L, p_L, rho_R, u_R, p_R, xc);
		exact_->initialize();
	} else if (idx_problem_ == 2) { // Strong Lax
		rho_L = 1.0; u_L = 0.0; p_L = 1000.0;
		rho_R = 1.0; u_R = 0.0; p_R = 0.01;
		xc = 0.5*(xl_+xr_);
	    for(int i = 0; i < N_cell_; i++) {
			if(x_[i] < xc) {
				rho0 = rho_L; u0 = u_L; p0 = p_L;
			} else {
				rho0 = rho_R; u0 = u_R; p0 = p_R;
			}
			qe_[0][i+gs_] = rho0;
			qe_[1][i+gs_] = u0;
			qe_[2][i+gs_] = p0;
			q_[0][i+gs_] = rho0;
			q_[1][i+gs_] = rho0*u0;
			q_[2][i+gs_] = 0.5*q_[1][i+gs_] * u0 + p0 / (GAMMA - 1.0);
		}
		bc_->setBC(open, open);
		te_ = 0.012;
		exact_->setProblem(rho_L, u_L, p_L, rho_R, u_R, p_R, xc);
		exact_->initialize();
	} else if (idx_problem_ == 3) { // shock turbulence interaction
	    for(int i = 0; i < N_cell_; i++) {
			if(x_[i] < 0.125*(xl_+xr_)) {
				rho0 = 3.857143;
				u0 = 2.629369;
				p0 = 10.333333;
			} else {
				rho0 = 1.0 + 0.2 * sin(50.0 * x_[i] - 25.0);
				u0 = 0.0;
				p0 = 1.0;
			}
			q_[0][i+gs_] = rho0;
			q_[1][i+gs_] = rho0*u0;
			q_[2][i+gs_] = 0.5*q_[1][i+gs_] * u0 + p0 / (GAMMA - 1.0);
		}
		bc_->setBC(open, open);
		te_ = 0.20;
	}

	if (IS_USE_T_END) te_ = T_END;
}

inline void Solver::updateDT() {
	double max_characteristic_speed = fluxer_->getMCS();
    dt_ = cfl_*dx_/(max_characteristic_speed + 1.0e-15);
}

void Solver::simpleSolve(unsigned int idx_problem, std::tuple<bool, std::string, int> preprocessed_data_info = {false, "./PreProcessedData/", 10}, bool is_logging_data = false) {	
	const bool is_make_preprocessed_data = std::get<0>(preprocessed_data_info);
	const std::string directory_of_preprocessed_data = std::get<1>(preprocessed_data_info);
	const int end_ts = std::get<2>(preprocessed_data_info);
	
	initProblem(idx_problem);

	showInfo();

	int max_substep = time_integral_->getMaxSubstep(), substep;
	//printf("address of q_bdry_: %p\n", q_bdry_);
	//printf("address of q_: %p\n", q_);

    while(t_ <= te_) {
		printf("ts = %d, t = %f, dt = %f\r", ts_, t_, dt_);
		if (is_make_preprocessed_data && ts_ <= end_ts && ts_ > 0) makePreProcessedData(directory_of_preprocessed_data+"problem"+std::to_string(idx_problem)+"_ts"+std::to_string(ts_)+".dat");
		substep = 1;

        do {
			time_integral_->setSubstep(substep);
			bc_->update(q_);
            recons_->update(q_bdry_, q_);
            fluxer_->update(flux_, q_bdry_);
            updateDT();
            time_integral_->update(q_, flux_);

			substep++;
        } while(substep <= max_substep);

		exact_->update(qe_);
        ts_ = ts_ + 1;
        t_ = t_ + dt_;
    }
	printf("\nSuccessfuly simulated!\n");

	printf("reconstruction time: %f\n", recons_->getReconstructionTime());

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << te_;
	std::string plot_file = "p"+std::to_string(idx_problem)+"_te"+stream.str();
	plot(plot_file.c_str());


	std::string log_file = "./PlotData/p"+std::to_string(idx_problem)+"_te"+stream.str()+".dat";  
	if (is_logging_data) write2File(log_file);
}

void Solver::plot(const char* file) const {
	FILE *fp;
	fp = popen("gnuplot", "w");

	double yM = -1.0e308, ym = 1.0e308;
	for (int i = 0; i < N_cell_; i++) {
		if (q_[0][i+gs_] > yM) yM = q_[0][i+gs_];
		if (q_[0][i+gs_] < ym) ym = q_[0][i+gs_];
	}
	int yM_quo = 0, ym_quo = 0;
	std::remquo(yM, 0.25, &yM_quo);
	std::remquo(ym, 0.25, &ym_quo);
	yM = 0.25*(yM_quo+1);
	ym = 0.25*(ym_quo-1);

	fprintf(fp, "set terminal png\n");
	fprintf(fp, "set output '%s.png' \n", file);
	fprintf(fp, "set xrange [%.2f:%.2f] \n", xl_, xr_); 
	fprintf(fp, "set yrange [%.2f:%.2f] \n", ym, yM); 
	fprintf(fp, "set xtics %.2f,0.5,%.2f  \n", xl_, xr_); 
	fprintf(fp, "set ytics %.2f,0.5,%.2f  \n", ym, yM);
	//fprintf(fp, "set xlabel \"{/=15 {/Arial-Italic x}}\" \n");
	//fprintf(fp, "set ylabel \"{/=15 {/Symbol-Oblique r}}\"  \n");
	fprintf(fp, "plot 0 title 'X-axis'lw 3.5 lc rgb 'black',"); 
	fprintf(fp, " '-'title 'Exact sol.' w line lw 2.5  lc rgb 'red',");
	//fprintf(fp, " '-'title 'Numerical sol(WENO3).' w point pt 7 ps 1 lc rgb 'blue',");
	fprintf(fp, " '-'title 'Numerical sol.' w point pt 7 ps 1 lc rgb 'blue'\n");
	/*
	for (int i = gs_; i <= N_cell_ - 1; i++) {
		fprintf(fp, "%f %f \n", x_[i], rhoe[i]); 
	}
	fprintf(fp, "e \n");
	for (int i = gs_; i <= N_cell_ - 1; i++) {
		if (selector_flag_rho[i] == 0) fprintf(fp, "%f %f \n", x[i], rho[i]); 
	}
	fprintf(fp, "e \n");
	for (int i = gs_; i <= N_cell_ - 1; i++) {
		if (selector_flag_rho[i] == 1) fprintf(fp, "%f %f \n", x[i], rho[i]); 
	}
	fprintf(fp, "e \n");
	*/	
	for (int i = 0; i < N_cell_; i++) {
		fprintf(fp, "%f %f \n", x_[i], qe_[0][i+gs_]); 
	}
	fprintf(fp, "e \n");
	for (int i = 0; i < N_cell_; i++) {
		fprintf(fp, "%f %f \n", x_[i], q_[0][i+gs_]);
		//fprintf(fp, "%f %f \n", x_[i], q_[i+gs_][1]/q_[i+gs_][0]); 
		//fprintf(fp, "%f %f \n", x_[i], (GAMMA - 1.0)*(q_[i+gs_][2] - 0.5*q_[i+gs_][1]*q_[i+gs_][1]/q_[i+gs_][0])); 
	}
	fprintf(fp, "e \n");
	fflush(fp);
}

void Solver::write2File(const std::string file) const {
	std::ofstream out(file);
	if (!out.is_open()) {
		std::cout  << "Fail to open:: " << file << std::endl;
		exit(EXIT_FAILURE); 
	}

	for (int i = 0; i < N_cell_; i++) {
		double rho = q_[0][i+gs_];
		double u = q_[1][i+gs_] / rho;
		double p = (GAMMA - 1)*(q_[2][i+gs_] - 0.5*rho*u*u);
		out << x_[i] << "\t" << rho << "\t" << u << "\t" << p
			<< "\t" << selector_[0][i+gs_] << "\t" << selector_[1][i+gs_] << "\t" << selector_[2][i+gs_]
			<< "\t" << qe_[0][i+gs_] << "\t" << qe_[1][i+gs_] << "\t" << qe_[2][i+gs_] << "\n";
	}

	out.close();
	std::cout  << file << " successfully saved. " << std::endl;
}

void Solver::makePreProcessedData(const std::string file) { // dataset have conservative variable data.
	std::ofstream out(file);
	if (!out.is_open()) {
		std::cout  << "Fail to open:: " << file << std::endl;
		exit(EXIT_FAILURE); 
	}

	const int N_variable = 3;
	double* q_pri;
	initVector1d<double>(q_pri, 5, 0.0);

	for (int i = 0; i < N_variable; i++) {
		for (int j = 0; j < N_cell_; j++) {
			if (i == 0) { // rho -> rho
				for (int k = 0; k < 5; k++) q_pri[k] = q_[0][j+gs_-2+k];
			}
			if (i == 1) { // rhou -> u
				for (int k = 0; k < 5; k++) q_pri[k] = q_[1][j+gs_-2+k]/q_[0][j+gs_-2+k];
			}
			if (i == 2) { // rhoE -> p
				for (int k = 0; k < 5; k++) q_pri[k] = (GAMMA - 1)*(q_[2][j+gs_-2+k] - 0.5*q_[1][j+gs_-2+k]*q_[1][j+gs_-2+k]/q_[0][j+gs_-2+k]);
			}

			int monotone_indicator = ((q_pri[2]-q_pri[1])*(q_pri[3]-q_pri[2]) < 0) ? 
										0 : 1;
			const int dummy = 0;
			out << x_[j] << "\t" << q_pri[0] << "\t" << q_pri[1] << "\t" << q_pri[2] << "\t" << q_pri[3] << "\t" << q_pri[4]
				<< "\t" << dummy << "\t" << dummy << "\t" << dummy << "\t" << dummy
				<< "\t" << monotone_indicator << "\t" << dummy << "\t" << dummy << "\t" << dummy << "\t" << selector_[i][j+gs_] << "\n";
		}
	}

	freeVector1d<double>(q_pri);
	out.close();
}

void Solver::showInfo() const {
	std::string name_of_recons = recons_->getName();
	std::string name_of_flux = fluxer_->getName();
	std::string name_of_timeintegral = time_integral_->getName();

	std::cout << "+------------------------------+ " << std::endl;
	std::cout << "|       Basic infomation       | " << std::endl;
	std::cout << "+------------------------------+ " << std::endl;
	std::cout << "# of cell : " << N_cell_ << std::endl;
	std::cout << "Computational field : [ " << xl_ << " : " << xr_ << " ]" << std::endl;
	std::cout << "Courant num. : " << cfl_ << std::endl;
	std::cout << "Simulation time : " << te_ << std::endl;
	std::cout << "Reconstruction : " << name_of_recons << std::endl;
	std::cout << "Riemann Solver : " << name_of_flux << std::endl;
	std::cout << "Time Integral : " << name_of_timeintegral << std::endl;
	std::cout << std::endl;
}