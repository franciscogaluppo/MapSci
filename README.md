# MapSci: compute research spaces
Source code for computing research spaces for prediction future research fields for scientists, institutions and states, using publication data.

## Evaluating the state-of-the-art in mapping research spaces: a Brazilian case study

### Abstract

Scientific knowledge cannot be seen as a set of isolated fields, but as a highly connected network. Understanding how research areas are connected is of paramount importance for adequately allocating funding and human resources (e.g., assembling teams to tackle multidisciplinary problems). The relationship between disciplines can be drawn from data on the trajectory of individual scientists, as researchers often make contributions in a small set of interrelated areas. Recent works by Guevara *et al.* and Chinazzi *et al.* propose methods for creating research maps from scientists' publication records. The former uses a frequentist approach to create a transition probability matrix, whereas the latter learns embeddings (vector representations). Surprisingly, these models were evaluated on different datasets and have never been compared in the literature. In this work, we compare both models in a systematic way, using a large dataset of publication records from Brazilian researchers. We evaluate these models' ability to predict whether a given entity (scientist, institution or region) will enter a new field w.r.t. the area under the ROC curve. Moreover, we analyze how sensitive each method is to the number of publications and the number of fields associated to one entity. Last, we conduct a case study to showcase how these models can be used to characterize science dynamics in the context of Brazil.

|![subfields](Figures/tiff/Fig4.tif)|
|:-----:|
|Fraction of subfields in each of the intermediate fields in which some of the institutions are specialized (data from the time interval [2000, 2014]). Intermediate classifications are grouped and colorcoded according to the macro field.|

**Read the paper**: [*soon...*]()

## Our Data

[Brazilian Scientific Publication Records and Author Affiliations from Lattes until Feb 2017 (Anonymized)](https://zenodo.org/record/4288583)

This file contains anonymized data about researcher profiles and publication records available extracted from the Lattes Platform in in February 2017 using the LattesDataXplorer tool. Lattes is a vast repository of researchers' curriculum vitae, widely adopted in Brazil. This platform is maintained by the Brazilian National Council of Scientific and Technological Development (CNPq) and is an internationally renowned initiative.

## Acknowledgments

We used the code from [Curved Bezier edges in NetworkX](https://github.com/beyondbeneath/bezier-curved-edges-networkx) for drawing the edges of our graphs. Thanks!
