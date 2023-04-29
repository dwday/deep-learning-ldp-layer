# A custom TensorFlow layer for local derivative patterns 
[Akgun, Devrim. "TensorFlow based deep learning layer for Local Derivative Patterns." Software Impacts 14 (2022): 100452 https://doi.org/10.1016/j.simpa.2022.100452](https://www.sciencedirect.com/science/article/pii/S2665963822001361)

## Example usages:
### Separately:
  x1 = LDP(mode='single', alpha='0')(x1)    
  x2 = LDP(mode='single',alpha='45')(x2)    
  x3 = LDP(mode='single',alpha='90')(x3)    
  x4 = LDP(mode='single',alpha='135')(x4)   
### Mean of LDP 0, LDP 45, LDP 90,and LDP 135:   
  x = LDP(mode='mean')(x)   
### Separate features:   
  x = LDP(mode='multi')(x)    
  
## Example test model that uses four directions:
![alt text](images/model1.png)


## Using LDP for feature extraction:
![alt text](images/cifar10.png)
##  Mean LDP features:
![alt text](images/ldp_combined.png)
##  Mean LDP 0 features:
![alt text](images/ldp_0.png)
##  Mean LDP 45 features:
![alt text](images/ldp_45.png)
##  Mean LDP 90 features:
![alt text](images/ldp_90.png)
##  Mean LDP 135 features:
![alt text](images/ldp_135.png)

@article{akgun2022tensorflow,
  title={TensorFlow based deep learning layer for Local Derivative Patterns},
  author={Akgun, Devrim},
  journal={Software Impacts},
  volume={14},
  pages={100452},
  year={2022},
  publisher={Elsevier}
}
