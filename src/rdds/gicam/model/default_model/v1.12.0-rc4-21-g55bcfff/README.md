# GICAM v1.12.0-rc4-21-g55bcfff

Trained om MIVMIR v1.12.0-rc4 using 300p case patient data.

Changes to previous GICAM model is that MTAF module is removed
from genmod scoring. This was necessary to make GENMOD run
successfully in the `raredisease` `rank_variants` subworkflow
that contains only SNVs.

`raredisease`:  
```
* 1360ffe (HEAD -> master) Remove CI/CD, .github/
* 1716bc2 Filter trios for patient variants
* abbc355 makegicamdata: Collect GICAM column
* a9caf3c Add GICAM module
* 1618d4f Add MIVMIR inference module and create GICAM training dataset
```
