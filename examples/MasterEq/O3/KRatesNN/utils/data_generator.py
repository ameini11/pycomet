import re
import os
import numpy as np
import pandas as pd

from const import const



class DataGenerator(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        self.header   = '[PrODE - Data Gen]: '
        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Data Quantities -----------------------------------------------------
        self.dim   = inputs.dim                                     # Data dimensionality
        self.T_str = list(map(lambda x: str(int(x)), inputs.T))     # Temp. in string form

        # Retrieving Molecular data -------------------------------------------
        self.levels_file = inputs.database_path + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = inputs.database_path + '/grouping/O3_UMN/O2/LevelsMap_' + inputs.grouping + '.csv'
        self.g, self.EeV, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = inputs.database_path + '/kinetics/O3_UMN_' + inputs.grouping + '/'
        self.react_file = {'diss': 'Diss_Corrected.dat', 'exch': 'Exch_Type1.dat', 'inel': 'Inel.dat'}

        # Building Master Equation matrices -----------------------------------
        self.num_exc    = int((self.num_bins-1)*self.num_bins/2)
        self.num_diss   = int(self.num_bins)
        self.dim['out'] = self.num_exc + self.num_diss
        self.rates      = self.get_rates()


    ###########################################################################
    ### Training Data
    ###########################################################################
    def training_data(self):

        x = np.zeros((len(self.T_str), self.dim['inp']), dtype=np.float64)
        y = np.zeros((len(self.T_str), self.dim['out']), dtype=np.float64)

        for i, T_i in enumerate(self.rates):
            x[i,0] = float(T_i)
            y_diss = self.rates[T_i]['diss']
            y_exc  = self.rates[T_i]['exch'] + self.rates[T_i]['inel']
            y[i,:] = np.hstack((y_diss,y_exc))
        return x,y


    ###########################################################################
    ### Master Equation Rates
    ###########################################################################
    def get_rates(self):
        '''Retrieve rates coefficients from file.'''
        rates_coeff = dict()
        for T_i in self.T_str:
            if T_i in rates_coeff: continue
            rates_coeff[T_i] = dict()
            print(self.header + ' Temp.: %5d K' % int(T_i))
            for reac in self.react_file:
                data_file = self.rates_path + 'T' + T_i + 'K/' + self.react_file[reac]
                rates_coeff[T_i][reac] = self.read_rates(data_file, reac)
        return rates_coeff

    def read_rates(self, file, reac_name):
        '''Read rates coefficients from file.'''
        print(self.header + '   Reading', reac_name, 'rates from file')

        shape = (1,self.num_diss) if reac_name == 'diss' else (1,self.num_exc)
        rates = np.zeros(shape, dtype=np.float64)

        infile = open(file, 'r')
        for i, line in enumerate(infile):
            reac_eqn, rate_str = line.split(':')
            rate = float(rate_str.split(',')[0]) * 1.e-6

            if reac_name == 'diss':
                bins_idx = re.findall(r"\((.*?)\)",reac_eqn)
                rates[0,int(bins_idx[0])-1] = rate
            else:
                rates[0,i] = rate

        infile.close()
        return rates


    ###########################################################################
    ### Util Functions
    ###########################################################################
    def read_levels(self):
        '''Retrieve all the ro-vib levels.'''
        data = pd.read_csv(self.levels_file, header=None, skiprows=15, delim_whitespace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn  = np.array(data[0])                    # Vibrational Q.N.
        jqn  = np.array(data[1])                    # Rotational  Q.N.
        g    = 1.5*(2.0*jqn+1.0)                    # Degeneracy (Including Electronic One)
        EeV  = np.array(data[2]) * const.Eh_to_eV   # Energy in eV
        EeV  = EeV - EeV[0]
        num  = len(vqn)
        return g, EeV, num

    def read_groups(self):
        '''Retrieve mapping of each rot-vib level to the respective bin.'''
        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx  = np.array(data[0])                                                                           
        bins = np.array(data[1]) - 1    # Level-to-Bin
        num  = max(bins) + 1
        return bins, num
