# lipid-nanoparticle-screening

Ionizable lipid and lipid nanoparticle screening for efficient nucleic acid delivery

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

  Lipid nanoparticles are promising delivery mechanisms for a variety of drugs into the cytoplasm.[1] They have been employed to deliver a wide range of nucleic acids, most notably the mRNA in Pfizer’s and Moderna’s COVID-19 vaccines.[2] The main component and most important factor in improving delivery efficiency are the ionizable lipids.[3] These lipids are positively charged during the lipid nanoparticle formulation, but are neutral at physiological pH. This positive charge under acidic pH values promotes attraction with the negatively charged nucleic acid backbone, hence improving encapsulation efficiency.[4] Moreover, protonation aids with endosomal escape, since the protonated ionizable lipid interacts with the anionic endosomal membrane.[5] However, it is essential that the lipids remain neutral at physiological levels to avoid toxicity and immunogenicity, and to therefore increase circulation time and cellular uptake.[6] Hence, there is a narrow interval of acceptable pKas, usually between 6.1 and 6.7. [7, 8]
    Although lipids are designed rationally, they must still be synthesized to be screened, a process that can take months and lots of pricey reagents. Our project would allow for a more efficient screening of ionizable lipids by creating a model to predict their pKa, encapsulation efficiency, size and polydispersity index. Hence, researchers would only need to synthesize the lipids that the model considers a hit, greatly reducing the time and resources spent on low-quality lipids. 
    Firstly, a publicly available database was created from patents and publication. Afterwards, numerous supervised learning approaches were explored. 

[1] Alfutaimani, A. S., Alharbi, N. K., S Alahmari, A., A Alqabbani, A., & Aldayel, A. M. (2024). Exploring the landscape of Lipid Nanoparticles (LNPs): A comprehensive review of LNPs types and biological sources of lipids. International Journal of Pharmaceutics: X, 8(100305), 100305. https://doi.org/10.1016/j.ijpx.2024.100305
[2]Hou, X., Zaks, T., Langer, R., & Dong, Y. (2021). Lipid nanoparticles for mRNA delivery. Nature Reviews. Materials, 6(12), 1078–1094. https://doi.org/10.1038/s41578-021-00358-0
[3]Han, X., Zhang, H., Butowska, K., Swingle, K. L., Alameh, M.-G., Weissman, D., & Mitchell, M. J. (2021). An ionizable lipid toolbox for RNA delivery. Nature Communications, 12(1), 7233. https://doi.org/10.1038/s41467-021-27493-0 
[4]Schober, G. B., Story, S., & Arya, D. P. (2024). A careful look at lipid nanoparticle characterization: analysis of benchmark formulations for encapsulation of RNA cargo size gradient. Scientific Reports, 14(1), 2403. https://doi.org/10.1038/s41598-024-52685-1 
[5]Semple, S. C., Akinc, A., Chen, J., Sandhu, A. P., Mui, B. L., Cho, C. K., Sah, D. W. Y., Stebbing, D., Crosley, E. J., Yaworski, E., Hafez, I. M., Dorkin, J. R., Qin, J., Lam, K., Rajeev, K. G., Wong, K. F., Jeffs, L. B., Nechev, L., Eisenhardt, M. L., … Hope, M. J. (2010). Rational design of cationic lipids for siRNA delivery. Nature Biotechnology, 28(2), 172–176. https://doi.org/10.1038/nbt.1602 
[6]Schober, G. B., Story, S., & Arya, D. P. (2024). A careful look at lipid nanoparticle characterization: analysis of benchmark formulations for encapsulation of RNA cargo size gradient. Scientific Reports, 14(1), 2403. https://doi.org/10.1038/s41598-024-52685-1 
[7]Jayaraman, M., Ansell, S. M., Mui, B. L., Tam, Y. K., Chen, J., Du, X., Butler, D., Eltepu, L., Matsuda, S., Narayanannair, J. K., Rajeev, K. G., Hafez, I. M., Akinc, A., Maier, M. A., Tracy, M. A., Cullis, P. R., Madden, T. D., Manoharan, M., & Hope, M. J. (2012). Maximizing the potency of siRNA lipid nanoparticles for hepatic gene silencing in vivo. Angewandte Chemie (International Ed. in English), 51(34), 8529–8533. https://doi.org/10.1002/anie.201203263 
[8]Simonsen, J. B., & Larsson, P. (2025). A perspective on the apparent pKa of ionizable lipids in mRNA-LNPs. Journal of Controlled Release: Official Journal of the Controlled Release Society, 384(113879), 113879. https://doi.org/10.1016/j.jconrel.2025.113879 



## Installation
The following dependencies were used:
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

The environment must have all the necessary dependencies. In our case, rd-kit was installed using conda, and the project was carried out in this conda environment. For more information on rd-kit installation, pelase refer to the official guide: https://www.rdkit.org/docs/Install.html 


## Quick Start

import master_functions.py

run_knn_analysis(filepath: str) # to run the K-nearest-neighbors analysis

run_gbr_analysis(filepath: str) #to run the multioutput gradient descent analysis

run_gbr_mo_lr_analysis(filepath: str) #to run the multioutput gradient descent with a linear regression and linear transformation analysis

run_rf_analysis(filepath: str) # to run both regular and multi output randon forest


## Usage

Add detailed information and examples on how to use the project, covering its major features and functions.

```python

```

The run_XX_analysis function run the whole abalysis for the model they specify. They call on dataset_preparation to open the csv file, specify X and y, normalize the y, get rid of NaN, split on training/testing. More importantly, this function cals on smiles_to_ECFP to transform th SMILES int ecfp so it can be fed into the models. The general structure is XX.fit, XX.predict, calculate R^2 and MSE and plot.  


## Contribute

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request. See the [contribution guidelines](CONTRIBUTING.md) for more information.

## Support

If you have any issues or need help, please open an [issue](https://github.com/hackbio-ca/demo-project/issues) or contact the project maintainers.

## License

This project is licensed under the [MIT License](LICENSE).
