# Question 1
## Numerical Results
- 1 <br/>
call option price: 5.876024233827607  
put option price: 5.377272153095845
- 2 <br/>
call option price: 2.33942
0513720004  
put option price: 11.790793224915063
- 3 <br/>
call option price: 8.433318690109608  
put option price: 7.438302065026413
- 4 <br/>
call option price: 8.677645562336004  
put option price: 8.178893481604248
- 5 <br/>
call option price: 6.120654113455842  
put option price: 5.1256374883726465<br/>  

## Comments
The **call option** price is positively correlated to maturity, volatility and risk free rate but negatively correlated to strike price;  
The **put option** price is positively correlated to strike price, maturity and volatility, but negatively correlated to risk free rate.
## Source Code

# Question 2
## 2.1
To calculate the covariance of X and Z, we need the value of the mean and variance of Z.  
For X and Y are two independent standard normal random variables, 

$$
\mathbb{E}(X)= \mathbb{E}(Y)=0 \\
Var(X)=Var(Y)=1\\
Cov(X,Y)=0
$$

- Mean of Z

$$
\mathbb{E}(Z) = \mathbb{E}(\rho X + \sqrt{1-\rho^2}Y) = \rho \mathbb{E}(X) +\sqrt{1-\rho^2}\mathbb{E}(Y)
$$

- Variance of Z

$$
Var(Z) = Var(\rho X + \sqrt{1-\rho^2}Y)=\rho^2Var(X)+(1-\rho^2)Var(Y)=1
$$

- Value of XZ

$$
XZ = \rho X^2 +\sqrt{1-\rho^2}XY \\ 
$$

- Mean of XZ

$$
\mathbb{E}(XZ) = \rho\mathbb{E}(X^2)+\sqrt{1-\rho^2}\mathbb{E}(XY) \\ 
=\rho(Var(X)+\mathbb{E}(X)^2) + \sqrt{1-\rho^2}\mathbb{E}(XY) \\
= \rho(Var(X)+\mathbb{E}(X)^2) + \sqrt{1-\rho^2}(Cov(X,Y)+\mathbb{E}(X)\mathbb{E}(Y)) 
$$

- Covariance of X and Z

$$
Cov(X,Z)=\mathbb{E}((X-\mathbb{E}(X))(Z-\mathbb{E}(Z)))
= \mathbb{E}(XZ) - \mathbb{E}(X)\mathbb{E}(Z) \\
= \rho(Var(X)+\mathbb{E}(X)^2) + \sqrt{1-\rho^2}\mathbb{E}(XY)-\mathbb{E}(X)(\rho \mathbb{E}(X) +\sqrt{1-\rho^2}\mathbb{E}(Y))\\
= \rho Var(X)+\sqrt{1-\rho^2}Cov(X,Y)
$$

- $\rho$(X,Z)  
To avoid the same notation, denote the correlation coefficient of X and Z as corr(X,Z)

$$
Corr(X,Z) = \frac{Cov(X,Z)}{\sqrt{Var(X)Var(Z)}} \\
= \frac{\rho Var(X)+\sqrt{1-\rho^2}Cov(X,Y)}{\sqrt{Var(X)[\rho^2Var(X)+(1-\rho^2)Var(Y)]}}\\
=\frac{\rho}{\sqrt{1}}=\rho
$$



## 2.2
