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
