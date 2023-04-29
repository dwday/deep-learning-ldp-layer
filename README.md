# A custom TensorFlow layer for local derivative patterns 
## Example usages:
### Separately:
  x1 = LDP(mode='single', alpha='0')(x1)
    
  x2 = LDP(mode='single',alpha='45')(x2)
    
  x3 = LDP(mode='single',alpha='90')(x3)
    
  x4 = LDP(mode='single',alpha='135')(x4)
   
### Combined:   
  x = LDP(mode='multi')(x)
    
## Example test model that uses four directions:
![alt text](images/model1.png)


[Akgun, Devrim. "TensorFlow based deep learning layer for Local Derivative Patterns." Software Impacts 14 (2022): 100452.. DOI:10.22531/muglajsci.830691](https://www.sciencedirect.com/science/article/pii/S2665963822001361)

@article{akgun2022tensorflow,
  title={TensorFlow based deep learning layer for Local Derivative Patterns},
  author={Akgun, Devrim},
  journal={Software Impacts},
  volume={14},
  pages={100452},
  year={2022},
  publisher={Elsevier}
}
