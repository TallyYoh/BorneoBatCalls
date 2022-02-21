
#                                                   Due to be released soon!



## Bats of Borneo: semi-automated classifier for echolocation calls 🦇
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4725680.svg)](https://doi.org/10.5281/zenodo.4725680)
#### Semi-automated acoustic classifier for the bats of Borneo for R, see https://doi.org/10.1016/j.ecolind.2022.108696 for more information 

Yoh, N., Kingston, T., McArthur, E., Aylen, O.E., Huang, J.C.C., Jinggong, E.R., Khan, F.A.A., Lee, B.P.Y.H., Mitchell, S.L.M., Bicknell, J.E.,
and Struebig, M.J. (2022). A machine learning framework to classify Southeast Asian echolocating bats, Ecological Indicators, 136.
doi:10.1016/j.ecolind.2022.108696 
​

This script applies the Borneo Bat Classifier (BBC) machine learning classifier to collated bat call parameter measurements from Borneo and assigns relevant labels
#### Data inputs required:
###### - WAV files in 5 second fragments (users)
###### - Machine learning models (available to download - "model_Type_1000.rds", "model_CF_1000.rds", "model_FMqCF_2000.rds")
###### - Confidence thresholds for each species (available to download - "ThresholdValues.csv")

#### The output includes: 
###### - Pulse measurements 
###### - Predicted class labels (call type/sonotype/species) 
###### - Confidence of class label - Script to locate files based on species & confidence on your desktop & reorder them for manual verification

#### Software used to create classifier
###### - R v3.6.3
###### - Kaleidoscope v5.1.9g (Wildlife Acoustics, 2019)
###### - Adobe Audition v12.1.5 (Adobe Creative Cloud)

#### Our aim is continuously test and update this tool as new reference data becomes available. Therefore, we would greatly appreciate users sharing any issues they find, particularly if this relates to species' IDs. Thank you!

#### Funding
NY was funded by a Natural Environmental Research Council (NERC) EnvEast DTP scholarship (grant number NE/L002582/1). NERC also funded the acoustic surveys in Sabah (NE/K016407/1; https://lombok.nerc-hmtf.info/) along with the Mohamed bin Zayed species Conservation Fund (11253049). FAAK and ERJ were supported by the Ministry of Higher Education Malaysia through F07/FRGS/1878/2019. JCCH was supported by the Global Biodiversity Information Facility through BIFA grant. BPYHL was supported by a Ministry of National Development EDGE Scholarship and the Wildlife Reserves Singapore Conservation Fund. TK was funded by the US National Science Foundation (165871). We thank the Sabah Biodiversity Council, Sabah Forest Department, Yayasan Sabah, and Benta Wawasan Sdn Bhd. for research permissions in Sabah. 

