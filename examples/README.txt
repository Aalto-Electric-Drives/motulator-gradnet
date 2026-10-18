Examples
========

These examples train GradNet current and flux linkage maps from measured and FEM datasets and compare the trained maps with the data. The datasets in ``examples/datasets`` are for two PM synchronous reluctance machines: a four-pole 5.6-kW machine (Baldor ECS101M0H7EF4) and a 10-pole 28-kW machine (Brusa HSM1.10.18.04). For the 5.6-kW machine, a FEM dataset with spatial harmonics is also provided. The trained models in ``examples/trained_models`` are loaded if they exist, so the examples run without training; delete a model file to retrain it. The scripts ``train_*.py`` train the FEM-data models used in the simulation examples of motulator (which has its own copies of the trained models it uses); they produce no figures and are shown without being executed.
