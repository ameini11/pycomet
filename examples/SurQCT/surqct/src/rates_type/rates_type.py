import importlib

from surqct.src.utilities import utils


# RatesType
###############################################################################
class RatesType():

    def __init__(
        self,
        System,
        InputData
    ):
        self.System = System
        self.InputData = InputData
        
        if self.System.Processes.ProcBool[0] > 0:
            self.rates_type = 'diss'
        else:
            self.rates_type = 'excit'

        self.rates_type_mod = importlib.import_module(
            f'surqct.src.rates_type.{self.System.NAtoms}_atoms.{self.rates_type}'
        )

    # Generate data
    def generate_trainingdata(self, *args, **kwargs):
        attr = 'generate_trainingdata'
        if hasattr(self.rates_type_mod, attr):
            return self.rates_type_mod.generate_trainingdata(
                System=self.System,
                InputData=self.InputData,
                *args,
                **kwargs
            )
        else:
            utils.raise_value_err(
                f"Rate types module has not '{attr}' attribute."
            )

    def generate_testingdata(self, *args, **kwargs):
        attr = 'generate_testingdata'
        if hasattr(self.rates_type_mod, attr):
            return self.rates_type_mod.generate_testingdata(
                System=self.System,
                InputData=self.InputData,
                *args,
                **kwargs
            )
        else:
            utils.raise_value_err(
                f"Rate types module has not '{attr}' attribute."
            )

    def generate_predictiondata(self, *args, **kwargs):
        attr = 'generate_predictiondata'
        if hasattr(self.rates_type_mod, attr):
            return self.rates_type_mod.generate_predictiondata(
                System=self.System,
                InputData=self.InputData,
                *args,
                **kwargs
            )
        else:
            utils.raise_value_err(
                f"Rate types module has not '{attr}' attribute."
            )

    # Save predicted data
    def save_predictiondata(self, *args, **kwargs):
        attr = 'save_predictiondata'
        if hasattr(self.rates_type_mod, attr):
            return self.rates_type_mod.save_predictiondata(
                System=self.System,
                InputData=self.InputData,
                *args,
                **kwargs
            )
        else:
            utils.raise_value_err(
                f"Rate types module has not '{attr}' attribute."
            )

    # Plotting
    def plot_prediction2D(self, *args, **kwargs):
        attr = 'plot_prediction2D'
        if hasattr(self.rates_type_mod, attr):
            return self.rates_type_mod.plot_prediction2D(
                System=self.System,
                InputData=self.InputData,
                *args,
                **kwargs
            )
        else:
            utils.raise_value_err(
                f"Rate types module has not '{attr}' attribute."
            )

    def plot_prediction3D(self, *args, **kwargs):
        attr = 'plot_prediction3D'
        if hasattr(self.rates_type_mod, attr):
            return self.rates_type_mod.plot_prediction3D(
                System=self.System,
                InputData=self.InputData,
                *args,
                **kwargs
            )
        else:
            utils.raise_value_err(
                f"Rate types module has not '{attr}' attribute."
            )
