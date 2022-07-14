# Pipelines
**Pipelines for pre-/post-processing of the SurQCT model**


## Workflow

1. **CoarseAir**

    Not used yet.
    
2. **SurQCT Model**

    Construct your SurQCT model and write the rates in HDF5 format.

3. **PyCoarseAir**

    In `PyCoarseAir/Exec` modify `PyCoarseAIR.py` accordingly and run it as follows
    ```
    $ python3 PyCoarseAIR.py <path-to-folder>
    ```
    It produces the database for PLATO (`kinetics` and `thermo`) with `<path-to-folder>` containing the `InputData.py` file accordingly modified.

4. **Plato**

    In `Plato/Exec` modify `Main` accordingly and run it as follows
    ```
    $ ./Main
    ```
    to run the Master Equation.

5. **MCoarseAir**

    In `MCoarseAir/Exec` modify appropriately the main file correspoinfg the chemical system you are studing and run it as follows
    ```
    $ ./matlab <main-file>
    ```
    to postprocess the Master Equation results.
