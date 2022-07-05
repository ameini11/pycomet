def get_rates_type_mod(System):
    if System.ThermochemicalProc.ProcBool[0]:
        RatesType = 'diss'
    else:
        RatesType = 'excit'
    return __import__(
        f'surqct.src.rates_type.{System.NAtoms}_atoms.{RatesType}',
        globals(),
        locals(),
        [RatesType],
        0
    )