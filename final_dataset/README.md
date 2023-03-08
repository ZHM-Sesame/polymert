# Dataset
This directory contains the final datasets that we plan to use for the project, with details as follows:
|Dataset         |Schema                         |Count                        |
|----------------|-------------------------------|-----------------------------|
|Copolymers		 |`monoA,monoB,fracA,fracB,chain_arch,property`|85932          |
|Homopolymers    |`mono,property,value`        |738            			   |
|Monomer-Fingerprint-Mappings |`mono,fingerprint (1024)`|1053|

- **Copolymers**: Contains the data about copolymers, with 3 different chain_architectures ('alternating', 'block', 'random') and 3 different compositions (1:1, 1:2, 2:1).
- **Homopolymers**: Contains the data about copolymers, with 3 different chain architectures ('alternating', 'block', 'random') and 3 different compositions (1:1, 1:2, 2:1).
- **Monomer-Fingerprint-Mappings**: As there were only 1053 unique monomers in the 2 previously mentioned datasets, it made sense to preprocess their fingerprints and store them in a separate dataset which can be used as a dictionary in our project, saving a lot of compute time which would've gone into calculating the fingerprint values.

Both copolymers and homopolymers datasets have 2 different property values we plan to predict --- Electron Affinity and Ionization Potential.

Note: The script (Polymer-TL-Dataprep.ipynb) which was used to create these datasets has been included as well.

References: 
- The copolymer dataset was created from the Vipea dataset used by [Aldeghi and Coley](https://pubs.rsc.org/en/content/articlelanding/2022/SC/D2SC02839E). It can be found [here](https://github.com/coleygroup/polymer-chemprop-data/tree/main/datasets/vipea).
- The homopolymer dataset was created from the Georgia Tech's MTL_Khazana dataset found [here](https://khazana.gatech.edu/).  