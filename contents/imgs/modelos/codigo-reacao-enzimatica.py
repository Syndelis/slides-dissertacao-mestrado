import argparse, contextlib, sys, os
import scipy
import numpy as np



def initial_values() -> np.ndarray:
    E_0 = 15.0
    EI_0 = 0.0
    ES_0 = 0.05
    I_0 = 5.0
    P_0 = 0.0
    S_0 = 30.0
    return np.array((
        E_0,
        EI_0,
        ES_0,
        I_0,
        P_0,
        S_0,
        ))


def constants() -> list:
    k1 = 0.2
    k2 = 0.1
    k3 = 0.5
    k4 = 0.05
    k5 = 0.05
    return [
        k1,
        k2,
        k3,
        k4,
        k5,
        ]


def variable_names() -> list[str]:
    return [
        "E",
        "EI",
        "ES",
        "I",
        "P",
        "S",
        ]


def system(t: np.float64, y: np.ndarray, *constants) -> np.ndarray:
    # populations
    E,EI,ES,I,P,S, = y
    # constants
    k1,k2,k3,k4,k5, = constants
    
    dE_dt = ((ES*k2 )+- (k1*S*E ) )+(ES*k3 )+((EI*k5 )+- (E*I*k4 ) ) 
    dEI_dt = (EI*k5 )+- (E*I*k4 ) 
    dES_dt = - ((ES*k2 )+- (k1*S*E ) )+- (ES*k3 ) 
    dI_dt = (EI*k5 )+- (E*I*k4 ) 
    dP_dt = ES*k3 
    dS_dt = (ES*k2 )+- (k1*S*E ) 

    return np.array([dE_dt,dEI_dt,dES_dt,dI_dt,dP_dt,dS_dt])


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