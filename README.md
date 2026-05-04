# Non-myopic-multi-sensor-multi-Bernoulli-GOSPA-driven-sensor-management
Sensor management code for non-myopic, multi-sensor, multi-Bernoulli GOSPA driven sensor management. Utilising Monte Carlo Tree Search adapted to sensor management and utilising an upper bound on the mean square generalised optimal sub pattern assignment (GOSPA) error to drive the decision making.

This code base is for the centralised approachg to multi-sensor sensor management for multi-Bernoulli (MB) filtering.

The cost function used to drive the decision making is an upper bound on the mean square generalised optimal sub pattern assignment (GOSPA) error and the implementation is based on a Monte Carlo Tree Search (MCTS), adapted for the partially observable nature of sensor management. It is benchmarked against a non-myopic information theoretic sensor manager, namely the Kullback-Leibler divergence (KLD). 

To set the cost function for the sensor manager, a 'cost_function' parameter must be passed to the following methods:

- manage_multiple_sensors_mb_full_jointly_calculated_MCTS()
- manage_multiple_sensors_mb_individually_MCTS()

'KLD' and 'GOSPA' are the available implementations within this codebase.

Each sensor (S1, S2) has the ability to conduct its own decision making, independent of the other and the ability to share information to a centralised decision making module which jointly optimises the actions taken by the sensors.

The filtering is centralised, and all sensors operate on shared information from a centralised, multi-Bernoulli filtering loop, with a sequential update step, to avoid unnecessary information loss about the origin of the measurements recieved by either sensor.

The codebase is a PMB implementation with the Poisson elements intensity set to zero meaning that it is MB.

This is the code base used for the research and results used in the paper: G. Jones, A. F. García-Fernández, “GOSPA-Driven Non-Myopic Multi-Sensor Management with Multi-Bernoulli Filtering,” https://arxiv.org/abs/2511.01045



The predecessor papers to this codebase/research are: 

ISIF Fusion 2023 - https://ieeexplore.ieee.org/abstract/document/10224220

IEEE TAES 2024 - https://ieeexplore.ieee.org/abstract/document/10571358 or https://arxiv.org/abs/2405.05815

MFI 2024 - https://ieeexplore.ieee.org/abstract/document/10705781

Google Scholar:  https://scholar.google.com/citations?user=hgzQhO8AAAAJ&hl=en
