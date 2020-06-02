# Seismic interpretation
In this ML recipe, we demonstrate a simplified version of Seismic interpretation project, a recent development of Neuromation in the field of commercial oil exploration.   

### Introduction
Seismic interpretation is a method of assessing petrophysical features of the subsurface from well-log and seismic data. Petrophysical data such as radioactivity, porosity, acoustic impedance, density, conductivity etc. are being logged at specific well locations and depths. Seismic data is a 3D representation of the entire reservoir area. The goal of seismic interpretation is finding an estimator from seismic data to the sparsely available well-log data so that petrophysical data can be predicted from the seismic data for the entire reservoir volume. 

### Data
For demonstration purposes, we use open data from the Netherlands Offshore F3 Block, [link](https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete). F3 is a block in the Dutch sector of the North Sea. The block is covered by 3D seismic that was acquired to explore for oil and gas in the Upper-Jurassic - Lower-Cretaceous strata.  The characteristics of the 3D seismic survey are the following: 
- inline range: 100 through 750 lines, 25 m/line step
- crossline range: 300 through 1250 lines, 25 m/line step
- Z range: 0 though 1848 ms, 4 ms step
- area: 375.31 km^2
 
Within the survey, only four vertical wells are present. All wells have sonic and gamma ray logs. In this demo, we implement 4-fold crossvalidation training our model on 3 wells and validating on the remaining well.

![](./img/scene.png)

### Method
The model is 2D-2D transcoder network trained to convert a 2D seismic slice into 2D petrophysical image:

![](./img/network.png)

As input, we use multiple inlines/crosslines in the neighbourhood of the 4 wells. We use 21 nearest slices along each horizontal dimension. Training targets are being constructed as vertical projection of petrophysical data onto corresponding seismic slices. We transform 1D logs into 2D image by adding second dimension and expanding it to 20-30 traces. Obviously, these targets cover only small region of a seismic slice, and training is implemented with 2D masked Mean Squared Error (MSE) loss.
 
![](./img/data.png)

### Results

We crossvalidate Gamma log model on the dataset of 168 projections (4 wells x 2 horizontal directions x 21 slice in each direction). We minimize masked MSE loss for 150 epochs. The model typically converges after 50 epochs. The Pearson correlation between true and predicted well logs is remarkably high, especially taking into account the limited size of the training data:

|well|correlation|
|:---:|:---:|
|F02-1|0.88|
|F03-2|0.70|
|F03-4|0.82|
|F06-1|0.81|
|**average**|0.80|

![](./img/results.png)

Needless to say that 4 item dataset isn't enough to draw serious conclusions from any ML project. The goal of this recipe is to demonstrate potential of the method with open seismic data. The commercial project this study is based on reaches correlation of ~90% and low MSE using proprietary data with large number of wells of complex geometry and other petrophysical properties.

## Quick Start

Sign up at [neu.ro](https://neu.ro) and setup your local machine according to [instructions](https://neu.ro/docs).
 
Then run:

```shell
neuro login
make setup
make jupyter
```

See [Help.md](HELP.md) for the detailed Neuro Project Template Reference.
