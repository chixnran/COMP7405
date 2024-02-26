All the files can be found in https://github.com/chixnran/COMP7405/tree/main/Assignment2 for your convenience.
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
[code question 1](sourcecode.ipynb#Question-1)


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
Cov(X,Z)=\mathbb{E}((X-\mathbb{E}(X))(Z-\mathbb{E}(Z))) \\ 
\qquad = \mathbb{E}(XZ) - \mathbb{E}(X)\mathbb{E}(Z) \\
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
[code question 2.2](sourcecode.ipynb#Question-2)

# Question 3
## 3.1
[code question 3.1](sourcecode.ipynb#Question-3.1)
## 3.2
### 3.2.1
[code question 3.2.1](ourcecode.ipynb#Question-3.1#3.2.1)<br/>
[31.csv](31.csv)<br/>
[32.csv](32.csv)<br/>
[33.csv](33.csv)<br/>
### 3.2.2
[code question 3.2.2](ourcecode.ipynb#Question-3.1#3.2.2)<br/>
09:31:00
![31.png](attachment:31.png)

09:32:00
![32.png](attachment:32.png)

09:33:00
![33.png](attachment:33.png)

## 3.3

![%E6%88%AA%E5%B1%8F2024-02-26%2020.18.30.png](attachment:%E6%88%AA%E5%B1%8F2024-02-26%2020.18.30.png)

To check whether there exists arbitrage opportunity, i choose to use put-call parity formula. The parameters are set as follows: <br/>
- the volitility is set to be 0.3 considering that after computation there are large amount of NaN
- Call / Put price: when calculating the prices of option, use the predefined function IV<br/>

The column 'diff' stands for the difference between two assumed portfolio:<br/>

    left side: P1= call option + PV(K)<br/>
    right side: P2: put option + PV(asset price)<br/>

Trough the output dataframe, we can see that in the situation of no transaction cost, there are several arbitrage opportunity detected. If the difference is positive, the price of the portfolio of a call_price and Ke^(-r(T-t)) risk-free zero-coupon bond is higher than the put option and asset, which means that the portfolio 1 is overvalued and we can get risk free benefit from buying P2 and short selling P1.
