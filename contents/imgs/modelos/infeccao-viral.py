import argparse, contextlib, sys, os
import scipy
import numpy as np



def initial_values() -> np.ndarray:
    C_0 = 0.0
    D_0 = 0.0
    E_0 = 5000.0
    I_0 = 0.0
    T_0 = 10.0
    V_0 = 10.0
    return np.array((
        C_0,
        D_0,
        E_0,
        I_0,
        T_0,
        V_0,
        ))


def constants() -> list:
    a = 0.05
    beta = 0.05
    beta_C = 0.2
    constant_1 = 1.0
    constant_2 = 2.0
    k_E = 0.0004
    k_I = 0.05
    k_V = 0.2
    m_C = 0.3
    m_D = 0.1
    m_T = 0.3
    s_T = 0.8
    return [
        a,
        beta,
        beta_C,
        constant_1,
        constant_2,
        k_E,
        k_I,
        k_V,
        m_C,
        m_D,
        m_T,
        s_T,
        ]


def variable_names() -> list[str]:
    return [
        "C",
        "D",
        "E",
        "I",
        "T",
        "V",
        ]


def system(t: np.float64, y: np.ndarray, *constants) -> np.ndarray:
    # populations
    C,D,E,I,T,V, = y
    # constants
    a,beta,beta_C,constant_1,constant_2,k_E,k_I,k_V,m_C,m_D,m_T,s_T, = constants
    
    dC_dt = (C*beta_C )+- (m_C*C ) 
    dD_dt = (beta*((constant_2*I )+T ) )+(m_D*D ) 
    dE_dt = - (k_E*E*V ) 
    dI_dt = (k_E*E*V )+- (k_I*I*T )+- (a*I ) 
    dT_dt = (s_T*(constant_1+C ) )+- (m_T*T ) 
    dV_dt = (a*I )+- (k_V*V ) 

    return np.array([dC_dt,dD_dt,dE_dt,dI_dt,dT_dt,dV_dt])


def simulation_output_to_csv(sim_steps, simulation_output, write_to) -> str:
    if not simulation_output.success:
        print(simulation_output.message)
        return

    populatio_values_per_dt = simulation_output.y.T

    write_to.write(f"t,{','.join(variable_names())}\n")

    for dt, y in zip(sim_steps, populatio_values_per_dt):
        write_to.write(f"{dt},")
        write_to.write(",".join(f"{val:.4f}" for val in y))
        write_to.write("\n")


def plot_simulation(sim_steps, simulation_output, filename):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename) as pdf:
        # All
        all_fig, all_ax = plt.subplots()
        all_ax.grid()
        all_ax.set(title=__name__, xlabel="Time", ylabel="Concentration")

        # Individually
        for variable_name, variable_line_data in zip(variable_names(), simulation_output.y):
            fig, ax = plt.subplots()
            ax.grid()

            ax.set(
                title=variable_name,
                xlabel="Time",
                ylabel="Concentration",
            )

            ax.plot(simulation_output.t, variable_line_data)
            all_ax.plot(simulation_output.t, variable_line_data)

            pdf.savefig(fig)

        pdf.savefig(all_fig)


def file_or_stdout(filename: str | None):
    if filename:
        return open(filename, 'w')
    else:
        return sys.stdout


def simulate(filename, st=0, tf=10, dt=0.1, plot=False):
    sim_steps = np.arange(st, tf + dt, dt)

    simulation_output = scipy.integrate.solve_ivp(
        fun=system,
        t_span=(0, tf + dt),
        y0=initial_values(),
        args=constants(),
        t_eval=sim_steps,
    )

    if plot:
        plot_simulation(sim_steps, simulation_output, filename)

    else:
        with file_or_stdout(filename) as f:
            simulation_output_to_csv(sim_steps, simulation_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--st", type=float, default=0)
    parser.add_argument("--tf", type=float, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.output is None and not args.csv:
        parser.error("when plotting (a.k.a --no-csv), an output file name is required via --output")

    if args.output:
        dirs = os.path.dirname(args.output)

        if dirs:
            os.makedirs(dirs, exist_ok=True)

    simulate(args.output, plot=not args.csv)