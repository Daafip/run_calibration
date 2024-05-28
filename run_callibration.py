import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from datetime import timedelta
import xarray as xr
from tqdm import tqdm
import gc
from typing import Any
from ewatercycle.forcing import sources
from ewatercycle_DA import DA
from pydantic import BaseModel

param_names = ["Imax", "Ce", "Sumax", "Beta", "Pmax", "Tlag", "Kf", "Ks", "FM"]
stor_names = ["Si", "Su", "Sf", "Ss", "Sp"]


class Experiment(BaseModel):
    """Runs DA experiment"""

    n_particles: int
    storage_parameter_bounds: tuple
    s_0: Any
    model_name: str
    HRU_id: str
    experiment_start_date: str
    experiment_end_date: str
    alpha: float
    save: bool

    units: dict = {}
    time: list = []
    lst_N_eff: list = []
    lst_n_resample_indexes: list = []
    lst_camels_forcing: list = []

    p_initial: Any = None
    ensemble: Any = None
    paths: tuple[Path, ...] | None = None
    ref_model: Any = None
    ds_obs_dir: Path | None = None
    state_vector_arr: Any = None

    @staticmethod
    def H(Z):
        """Operator function extracts observable state from the state vector"""
        len_Z = 15
        if len(Z) == len_Z:
            return Z[-1]
        else:
            raise SyntaxWarning(f"Length of statevector should be {len_Z} but is {len(Z)}")

    @staticmethod
    def calc_NSE(Qo, Qm):
        Qo[Qo == 0] = 1e-8
        Qm[Qm == 0] = 1e-8
        QoAv = np.mean(Qo)
        ErrUp = np.sum((Qo - Qm) ** 2)
        ErrDo = np.sum((Qo - QoAv) ** 2)
        return 1 - (ErrUp / ErrDo)

    @staticmethod
    def calc_log_NSE(Qo, Qm):
        Qo[Qo == 0] = 1e-8
        Qm[Qm == 0] = 1e-8
        QoAv = np.mean(Qo)
        ErrUp = np.sum((np.log(Qo) - np.log(Qm)) ** 2)
        ErrDo = np.sum((np.log(Qo) - np.log(QoAv)) ** 2)
        return 1 - (ErrUp / ErrDo)

    def set_up_forcing(self):
        self.make_paths()
        forcing_path = self.paths[0]

        ### no longer needed likely
        # origin_camels_path = forcing_path / f'{self.HRU_id}_lump_cida_forcing_leap.txt'
        # lst_camels_file_path = []
        # for n in range(self.n_particles):
        #     camels_file_path = forcing_path / f'{self.HRU_id}_lump_cida_forcing_leap_{n}.txt'
        #     if not camels_file_path.is_file():
        #         shutil.copy(origin_camels_path, camels_file_path)
        #     lst_camels_file_path.append(camels_file_path.name)

        for n in range(self.n_particles):
            self.lst_camels_forcing.append(
                sources.HBVForcing(start_time=self.experiment_start_date,
                                   end_time=self.experiment_end_date,
                                   directory=forcing_path,
                                   camels_file=f'{self.HRU_id}_lump_cida_forcing_leap.txt',  # lst_camels_file_path[n]
                                   alpha=self.alpha,
                                   ))
        # del lst_camels_file_path
        # gc.collect()

    def make_paths(self):
        path = Path.cwd()
        forcing_path = path / "Forcing"
        observations_path = path / "Observations"
        output_path = path / "Output"

        self.paths = forcing_path, output_path, observations_path

        for path_i in list(self.paths):
            path_i.mkdir(exist_ok=True)

    def initialize(self):
        """Contains actual code to run the experiment"""

        p_min_initial, p_max_initial, s_max_initial, s_min_initial = self.storage_parameter_bounds

        # set up ensemble
        self.ensemble = DA.Ensemble(N=self.n_particles)
        self.ensemble.setup()

        # initial values
        array_random_num = np.array(
            [[np.random.random() for _ in range(len(p_max_initial))] for _ in range(self.n_particles)])
        p_initial = p_min_initial + array_random_num * (p_max_initial - p_min_initial)
        self.p_initial = p_initial

        # values which you
        setup_kwargs_lst = []
        for index in range(self.n_particles):
            setup_kwargs_lst.append({'parameters': ','.join([str(p) for p in p_initial[index]]),
                                     'initial_storage': ','.join([str(s) for s in self.s_0]),
                                     })

        # this initializes the models for all ensemble members.
        self.ensemble.initialize(model_name=[self.model_name] * self.n_particles,
                                 forcing=self.lst_camels_forcing,
                                 setup_kwargs=setup_kwargs_lst)

        del setup_kwargs_lst, p_initial, array_random_num
        gc.collect()

    def load_obs(self):
        forcing_path, output_path, observations_path = self.paths
        # create a reference model
        self.ref_model = self.ensemble.ensemble_list[0].model

        # load observations
        self.ds_obs_dir = observations_path / f'{self.HRU_id}_streamflow_qc.nc'
        if not self.ds_obs_dir.exists():
            ds = xr.open_dataset(forcing_path / self.ref_model.forcing.pr)
            basin_area = ds.attrs['area basin(m^2)']
            ds.close()

            observations = observations_path / f'{self.HRU_id}_streamflow_qc.txt'
            cubic_ft_to_cubic_m = 0.0283168466

            new_header = ['GAGEID', 'Year', 'Month', 'Day', 'Streamflow(cubic feet per second)', 'QC_flag']
            new_header_dict = dict(list(zip(range(len(new_header)), new_header)))
            df_q = pd.read_fwf(observations, delimiter=' ', encoding='utf-8', header=None)
            df_q = df_q.rename(columns=new_header_dict)
            df_q['Streamflow(cubic feet per second)'] = df_q['Streamflow(cubic feet per second)'].apply(
                lambda x: np.nan if x == -999.00 else x)
            df_q['Q (m3/s)'] = df_q['Streamflow(cubic feet per second)'] * cubic_ft_to_cubic_m
            df_q['Q'] = df_q['Q (m3/s)'] / basin_area * 3600 * 24 * 1000  # m3/s -> m/s ->m/d -> mm/d
            df_q.index = df_q.apply(lambda x: pd.Timestamp(f'{int(x.Year)}-{int(x.Month)}-{int(x.Day)}'), axis=1)
            df_q.index.name = "time"
            df_q.drop(columns=['Year', 'Month', 'Day', 'Streamflow(cubic feet per second)'], inplace=True)
            df_q = df_q.dropna(axis=0)

            ds_obs = xr.Dataset(data_vars=df_q[['Q']])
            ds_obs.to_netcdf(self.ds_obs_dir)
            ds_obs.close()
            del df_q, ds, ds_obs
            gc.collect()

    def initialize_da_method(self):
        print(f'init_da', end=" ")
        self.ensemble.set_state_vector_variables("all")

        # extract units for later
        state_vector_variables = self.ensemble.ensemble_list[0].variable_names


        for var in state_vector_variables:
            self.units.update({var: self.ref_model.bmi.get_var_units(var)})

    def assimilate(self):
        ## run!
        n_timesteps = int((self.ref_model.end_time - self.ref_model.start_time) /
                          self.ref_model.time_step)

        lst_state_vector = []
        try:
            for i in tqdm(range(n_timesteps)):
                self.time.append(pd.Timestamp(self.ref_model.time_as_datetime.date()))

                self.ensemble.update(assimilate=False)

                state_vector = self.ensemble.get_state_vector()
                lst_state_vector.append(state_vector)

                del state_vector
                gc.collect()

        except KeyboardInterrupt:  # saves deleting N folders if quit manually
            self.ensemble.finalize()

        self.ensemble.finalize()

        self.state_vector_arr = np.array(lst_state_vector)
        del lst_state_vector, self.ensemble
        gc.collect()

    def create_combined_ds(self):
        data_vars = {}
        for i, name in enumerate(stor_names):
            storage_terms_i = xr.DataArray(self.state_vector_arr[:, :, i].T,
                                           name=name,
                                           dims=["EnsembleMember","time"],
                                           coords=[np.arange(self.n_particles), self.time],
                                           attrs={
                                               "title": f"HBV storage terms data over time for {self.n_particles} particles ",
                                               "history": f"results from ewatercycle_HBV.model",
                                               "description": "Modeled values",
                                               "units": f"{self.units[name]}"})
            data_vars[name] = storage_terms_i

        for i, name in enumerate(["Q"]):
            storage_terms_i = xr.DataArray(self.state_vector_arr[:, :, -1].T,
                                           name=name,
                                           dims=["EnsembleMember","time"],
                                           coords=[np.arange(self.n_particles), self.time],
                                           attrs={
                                               "title": f"HBV Q value over time for {self.n_particles} particles ",
                                               "history": f"results from ewatercycle_HBV.model",
                                               "description": "Modeled values",
                                               "units": f"{self.units[name]}"})
            data_vars[name] = storage_terms_i

        # # for callibration params are constant so only store once
        for i, name in enumerate(param_names):
            storage_terms_i = xr.DataArray(self.state_vector_arr[0, :, i].T,
                                           name=name,
                                           dims=["EnsembleMember"],
                                           coords=[np.arange(self.n_particles)],
                                           attrs={
                                               "title": f"HBV param terms data over time for {self.n_particles} particles ",
                                               "history": f" results from ewatercycle_HBV.model",
                                               "description": "Modeled values",
                                               "units": f"{self.units[name]}"})
            data_vars[name] = storage_terms_i


        ds_combined = xr.Dataset(data_vars,
                                 attrs={
                                     "title": f"HBV storage & parameter terms data over time for {self.n_particles} particles",
                                     "history": f"Storage term results from ewatercycle_HBV.model",
                                     "n_particles": self.n_particles,
                                     "HRU_id": self.HRU_id,
                                 }
                                 )

        ds_obs = xr.open_dataset(self.ds_obs_dir)
        ds_observations = ds_obs['Q'].sel(time=self.time)

        ds_combined['Q_obs'] = ds_observations
        ds_combined['Q_obs'].attrs.update({
            'history': 'USGS streamflow data obtained from CAMELS dataset',
            'url': 'https://dx.doi.org/10.5065/D6MW2F4D'})

        # only save the best run for (storage) efficiency
        NSE = np.zeros(self.n_particles)
        log_NSE = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            NSE[i] = self.calc_NSE(ds_observations.to_numpy(), ds_combined["Q"].isel(EnsembleMember=i).to_numpy().copy())
            log_NSE[i] = self.calc_log_NSE(ds_observations.values, ds_combined["Q"].isel(EnsembleMember=i).to_numpy().copy())

        i_max_nse = NSE.argmax()
        i_max_log_nse = log_NSE.argmax()

        if i_max_nse == i_max_log_nse:
            ds_combined = ds_combined.isel(EnsembleMember=i_max_nse)
        else:
            ds_combined = ds_combined.isel(EnsembleMember=[i_max_nse, i_max_log_nse])

        dict_nse = dict(NSE_max=NSE.max(),
                        log_NSE_max=log_NSE.max(),
                        i_NSE_max=i_max_nse,
                        i_log_NSE_max=i_max_log_nse)
        ds_combined.attrs.update(dict_nse)



        current_time = str(datetime.now())[:-10].replace(":", "_")
        if self.save:
            forcing_path, output_path, observations_path = self.paths
            file_dir = output_path / (
                f'{self.HRU_id}_N-{self.n_particles}_'
                f'{current_time}.nc')
            ds_combined.to_netcdf(file_dir)

        ds_obs.close()
        del (self.time, ds_obs, self.lst_n_resample_indexes)

        gc.collect()

        return ds_combined

    def finalize(self):
        forcing_path = self.paths[0]
        # remove temp file once run - in case of camels just one file
        for index, forcing in enumerate(self.lst_camels_forcing):
            forcing_file = forcing_path / forcing.pr
            forcing_file.unlink(missing_ok=True)

        # catch to remove forcing objects if not already.
        try:
            self.ensemble.finalize()
            del self.ensemble.finalize
        except AttributeError:
            pass  # already deleted previously


def run_experiment(HRU_id_int: Any,
                   storage_parameter_bounds: tuple,
                   experiment_start_date: str,
                   experiment_end_date: str,
                   ) -> xr.Dataset | None:
    # """Contains iterables for experiment"""

    model_name = "HBVLocal"

    s_0 = np.array([0, 100, 0, 5, 0])
    n_particles = 500

    # Hyper parameters
    alpha = 1.26
    print_ending = "\n"  # should be \n to show in promt, can be removed here by changing to ' '
    returned = None

    HRU_id = f'{HRU_id_int}'
    if len(HRU_id) < 8:
        HRU_id = '0' + HRU_id

    save = True

    current_time = str(datetime.now())[:-10].replace(":", "_")
    experiment = Experiment(n_particles=n_particles,
                            storage_parameter_bounds=storage_parameter_bounds,
                            s_0=s_0,
                            model_name=model_name,
                            HRU_id=HRU_id,
                            experiment_start_date=experiment_start_date,
                            experiment_end_date=experiment_end_date,
                            alpha=alpha,
                            save=save,
                            )
    try:
        print(f'starting {HRU_id} at {current_time}', end="\n")
        experiment.set_up_forcing()

        print(f'init ', end=print_ending)
        experiment.initialize()

        print(f'load obs ', end=print_ending)
        experiment.load_obs()

        print(f'setup_state_vector', end=print_ending)
        experiment.initialize_da_method()

        print(f'assimilate ', end=print_ending)
        experiment.assimilate()

        print(f'output ', end=print_ending)
        ds_combined = experiment.create_combined_ds()
        ds_combined.close()
        del ds_combined
        gc.collect()


    except Exception as e:
        print(e)

    finally:
        print(f'cleanup ', end=print_ending)
        experiment.finalize()

    del experiment
    gc.collect()

    return returned



"""
Check list for a new experiment:

    - All values passed correctly from main -> run ->  experiment_run
        preferably don't change the actual values passed experiment_run:
        if needed refactor to list/tuple
    - preferably keep iterable set in main
    - add iterable in attrs 
    - Meaningful file path

"""


def main():
    """Main script"""
    forcing_path = Path.cwd() / "Forcing"
    HRU_ids = [path.name[0:8] for path in
               forcing_path.glob("*_lump_cida_forcing_leap.txt")]
    n_start_skip = 0
    n_end_skip = 0

    total_nruns = len(HRU_ids) - n_start_skip - n_end_skip
    avg_run_length = 0.2  # hr
    total_hrs = total_nruns * avg_run_length
    estimated_finish = datetime.now() + timedelta(hours=total_hrs)
    print(
        f'based on {total_nruns}run @ {avg_run_length}hrs/run = est.finish: {estimated_finish.strftime("%Y-%m-%d %H:%M")}')

    for index, HRU_id_int in enumerate(HRU_ids):
        if index < n_start_skip or index > (len(HRU_ids)-n_end_skip):
            pass
        else:
            # initial guess
            try:
                p_min_initial = np.array([0, 0.2, 40, .5, .001, 1, .01, .0001, 6])
                p_max_initial = np.array([8, 1, 800, 4, .3, 10, .1, .01, 0.1])

                s_max_initial = np.array([10, 250, 100, 40, 150])
                s_min_initial= np.array([0, 150, 0, 0, 0])

                storage_parameter_bounds = (p_min_initial,
                                            p_max_initial,
                                            s_max_initial,
                                            s_min_initial)

                # Run longer
                experiment_start_date = "1997-08-01T00:00:00Z"
                experiment_end_date = "2002-09-01T00:00:00Z"

                run_experiment(HRU_id_int,
                               storage_parameter_bounds,
                               experiment_start_date,
                               experiment_end_date,
                               )

            except Exception as e:
                print(e)



if __name__ == "__main__":
    gc.enable()
    main()
