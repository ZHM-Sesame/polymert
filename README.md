# polymertl


### Install requirements

```setup
conda create -n polymertl python
source activate polymertl
pip install -r requirements.txt
``` 

# Getting data
Details about the datasets are listed below. 
Preprocessing will be implemented later to clean the data. 

| Property                    | Size  |
|-----------------------------|-------|
| EA and IP                   | 90954 |
| Tg                          | 131   |
| EA vs SHE and IP vs SHE     | 43377 |

| File               | Representation        | Polymer Type                                           | Tg   | EA   | IP   | EA vs SHE | IP vs SHE | Size        |
|--------------------|-----------------------|--------------------------------------------------------|------|------|------|-----------|-----------|-------------|
| figotj dataset 1   | single trimer         | conjugated binary Homopolymer                          |      | y    | y    |           |           | 47988       |
| figotj dataset 2   | graphs with 19F NMR   | Six monomer types                                      |      |      |      | y         | y         | 411         |
| figotj dataset 3   | 2 Monomers with ratio | random and block polyhydroxyalkanoate copolymers        | y    |      |      |           |           | 131         |
| JACS               | 2 Monomers w/o ratio  | AB alternating co-polymers                             |      |      |      |           |           | 6345        |
| diblock-phrase     |                       | Diblock Copolymers                                      |      |      |      |           |           |             |
| Vipea              | 2 Monomers with ratio | Block/alternating                                      | y    | y    | y    | y         | y         | 42966       |
| Fullerene Exp.     |                       |                                                        |      |      |      |           |           | 1203        |
| MTL_Khazana        | Single Smile          | Homopolymer                                            | y    | y    |      |           |           | 6264(comb.) |
| PolymerGasMembraneML| Single Smile         | Homopolymer                                            |      |      |      |           |           | 776         |

## Downloading and extracting the dataset

To download and extract the collected dataset summarized above, use the `get_data_collect.sh` script provided in the repository. Follow these steps:

   ```bash
   #Grant execute permissions to the script:
   chmod +x get_data_collect.sh
   #run the script
   ./get_data_collect.sh
