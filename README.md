# FunctionalMixedEffects.jl

This package provides backend calculations for fitting functional mixed effects models. For the R package front-end, see the [FunctionalMixedEffects.Rpkg](https://github.com/wzhorton/FunctionalMixedEffects.Rpkg) repository.

---

## Functional Data Analysis Background

Functional data, also known as curve data or waveform data, represent an extension to traditional data analysis where observations are entire functions rather than individual numbers. Of course, recording an infinite number of values in one curve is impossible, but accounting for the functional structure of the data is still worthwhile. 

The book *Functional Data Analysis* by [Ramsay and Silverman (2005)](https://link.springer.com/book/10.1007/b98888) is a foundational reference in this space. Among all the techniques they develop for functional data analysis (FDA), this package will focus on splines, or rather B-splines. See the [B-spline Wikipedia page](https://en.wikipedia.org/wiki/B-spline) for more detailed information on spline construction. A B-spline representable function with $p$ basis functions is expressed by:

$$
f(x) = \sum_{j=1}^p \theta_j b_j(x)
$$

where $\theta_j$ is the coefficient for basis function $b_j$. In general the basis functions will depend on a grid of knot locations (with replication at the ends corresponding to the spline degree), but the default used here is an evenly spaced set of knots resulting in $p$ cubic spline basis functions. 

The main reason to use splines for functional modeling is dimension-reduction. Instead of having to keep track of $f(x)$ for every single possible $x$, we have reduced the problem to just figuring out the $p$ coefficients $\theta_1,\ldots,\theta_p$. In other words, we have effectively reduced the problem of functional analysis to vector analysis (or multivariate analysis). Other basis representation systems exist, but this package favors splines due to simplicity, speed, sparsity, and convention.

### B-spline model statement

Suppose $n$ curves are observed, labeled $i=1,\ldots,n$. Each curve has a vector of observation values, denoted $\underline{y_i}$, with length $m$. The hierarchical model used in this package to represent each curve as a B-spline is:

$$
\underline{y_i} \sim N_m(H\underline{\theta_i}, \sigma^2 I_m)
$$

where $\underline{\theta_i}$ is the $p$-dimensional B-spline coefficient vector, $H$ is the $(m\times p)$ B-spline design matrix, $\sigma^2$ is the error variance between the spline curve and the data curve, and $I_m$ is the $(m\times m)$ identity matrix. [Lang and Brezger (2004)](https://www.tandfonline.com/doi/abs/10.1198/1061860043010) provide a *P-spline* or penalized B-spline prior that makes it easier to select $p$:

$$
\underline{\theta_i} \sim N_p(\underline{0}, \tau^2 P^{-1})
$$

where $P$ is known as a first-order penalty matrix. See the original paper, [Telesca and Inoue (2008)](https://www.tandfonline.com/doi/abs/10.1198/016214507000001139), or [Horton et al. (2021)](https://www.tandfonline.com/doi/full/10.1080/00401706.2020.1841033) for examples of the exact matrix form, including minor adjustments for rank deficiency. Selecting $p$ can be challenging: too small and the fitted function is not flexible enough, but too large and overfit becomes a problem. The penalized B-spline addresses this by penalizing the second derivative, allowing users to choose quite large $p$, even $p>m$, without overfit.

## Mixed Model Background

Mixed models are a tool used to control for non-independent errors in a regression. For example, clinical trials can often involve giving treatment to an individual multiple times (repeated measures problem). In that setting, each treatment response is not perfectly independent; observations may be correlated within an individual. The effect of an individual, often called a *random effect*, is usually not of interest, but is important to control for so that the underlying regression assumptions are met and the true treatment effect, often called a *fixed effect* can be accurately studied. Mixed models offer a framework for simultaneously accounting for random effects and estimating fixed effects.

In a Bayesian setting, mixed models are handled via the priors. Suppose that the $n$-dimensional observation vector $\underline{y}$ contains all recorded values for some experiment and that the $(n\times p)$ matrix $X$ contains all fixed effect values (such as treatment labels) and that the $(n\times q)$ matrix $Z$ contains all random effect values (such as personal ID labels). The regression model is written as:

$$
\underline{y} \sim N_n(X\underline{\beta} + Z\underline{u}, \sigma^2I_n)
$$

where $\underline{\beta}$ is the vector of fixed effect coefficients and $\underline{u}$ is the vector of random effect coefficients. For priors, the fixed effects have $\underline{\beta} \sim N_p(\underline{\mu_0},\Sigma_0)$ for fixed mean and covariance. The random effects have $\underline{u} \sim N_q(\underline{0}, W)$ where the covariance matrix $W$ is also given some prior (Wishart, factored, etc.). The important difference in the priors is in the fact that $\Sigma_0$ is fixed while $W$ is random (and therefore suggestible by the data). A larger effect coming from $W$ suggests that the individual random effects have a significant effect on the outcomes, and therefore the fixed effects may be less reliable. 

There is a lot more to be said about mixed models (See the [Mixed Model Wikipedia page](https://en.wikipedia.org/wiki/Mixed_model) for loads of detail). This package is motivated primarily by a repeated measured perspective, and the template scripts found in the `FunctionalMixedEffects.Rpkg` repo support this. However, the model fitting code itself is somewhat general (within the functional framework) and users are free to design their own suitable mixed model.

## The Functional Mixed Effects Model

With background established, the purpose of this package is to fuse the two ideas of functional analysis and mixed models. The need is obvious: many exercise science and kinesiology research labs collect waveform data from subjects in the presence of various treatments and experimental conditions. Almost always several curves are taken from each participant, thus the desire to account for repeated measures.

The model structure implemented in this package can be summarized as follows: model each observed curve using the P-spline framework, then place a multivariate mixed model structure on the spline coefficients. In this way, each covariate, fixed and random, will have an interpretable functional effect on the outcome. The individual random effects will account for the fact that curves coming from a specific subject are more similar than those from another subject, and will appropriately adjust the certainty levels around the functional fixed effects of interest.

The model structure implemented in this pacakge can be written in several ways, with the most concise involving the [matrix normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution). While this is overkill for most applications, the matrix normal distribution effectively summarizes the model statement and the sampling scheme. The section below describes a looped version of the model (i.e. $i=1,\ldots,n$) which is clearer to understand, but lacks the implementation details embedded in the matrix version.

### Hierarchical Model Statement

Suppose we observe $n$ curves $\underline{y_i}$ for $i=1,\ldots,n$ each containing $m$ points. Organize these into columns, forming the $(m\times n)$ matrix $Y$. The data likelihood is given by:

$$
Y \sim MN_{mn}(H \Theta, \sigma^2 I_m, I_n)
$$

where $H$ is the $(m\times p)$ B-spline design matrix, $\Theta$ is the $(p\times n)$ matrix of spline coefficients where column $i$ contains the coefficients for curve $i$, $\sigma^2$ is the curve-to-spline variance, and $I_K$ dentoes the $K\times K$ indentity matrix. The prior distribution on $\Theta$ contains the mixed model form, and is given by:

$$
\Theta \sim MN_{pn}(B_fX_f + B_rX_r, \tau^2 P^{-1}, I_n)
$$

where $X_f$ contains the $(q_f\times n)$ design matrix (transpose) of fixed effects, $B_f$ is the $(p\times q_f)$ matrix of fixed effect coefficients, $X_r$ contains the $(q_r \times n)$ design matrix (transpose) of random effects, $B_r$ is the $(p\times q_r)$ matrix of random effect coefficients, and $P$ is the $(p\times p)$ penalty matrix. For priors, $B_f \sim MN_{pq_f}(0,v_0 P^{-1},I_{q_f})$ is used for some fixed $v_0$, and $B_r \sim MN_{pq_r}(B_c X_c, \lambda^2 P^{-1}, I_{q_r})$ is used for a random $\lambda^2$.

The mean structure for $B_r$ allows for more complex structure for the random effects. For example, a hierarchical centering structure could be used to produce identifiable estimates of grouped random effects (nested within subject). Of course, hierarchical centering can be cast as a sum-to-zero constraint within either the fixed or random effects component, so it is up to the user to construct a suitable model. In the more general sense, $B_c$ is the $(p\times q_c)$ matrix of global random effect coefficients and $X_c$ is the $(q_c \times q_r)$ design matrix mapping the random effects to their hierarchical centers. Finally, $B_c$ is assigned a prior $B_c \sim MN_{pq_c}(0, v_c P^{-1}, I_{q_c})$. For each of $\sigma^2$, $\tau^2$, and $\lambda^2$, an inverse-gamma prior is used for conjugacy.  

Note that reusing $P^{-1}$ is actually required for the posterior to be matrix normal, otherwise it become a more general reshaped vector normal with non-kronecker product covariance (the sum of kronecker products, actually). This is also true for using $I_n$ for both $Y$ and $\Theta$. Technically, a scaling factor is permitted, but the code is structured to require an exact match because of non-identifiability issues. As a rule of thumb, for any given likelihood-prior pair where both are matrix normal, the covariance matrix whose dimension is common to both must be equivalent (e.g. $Y$ and $\Theta$ both contain $n$ as a dimension, so the $n\times n$ covariance must be shared.)

### Sampling Scheme

This model was designed, as many Bayesian hierarchical models are, to be full of conditional conjugacy. Thus, this package uses Gibbs sampling to draw from the model posterior and ultimately perform inference. 

- Sampling the variance parameters $\sigma^2$, $\tau^2$, and $\lambda^2$ is straightforward: draw from an inverse-gamma conditional distribution where the rate depends on the total squared error. Similar relationships can be found in the [conjugate prior wiki page](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions).
- Sampling the spline parameters $\Theta$ can either be done individually or jointly. If done individually, the well known normal-normal regression conjugacy result can be used (see (Bayesian linear regression Wiki](https://en.wikipedia.org/wiki/Bayesian_linear_regression) page).
- The same procedure can be used to sample $U$ and $B$, which is a matrix extension to the conjugate normal-normal result. However, this result does not appear to be online, so it is provided below in a general form. To sample $U$, use the result on $\Theta - BX$, and to sample $B$, similarly use the result on $\Theta - UZ$.

## Matrix Normal Conjugacy Result

Suppose a $(p\times n)$ matrix $\Theta$ is modeled:

$$
\Theta \sim MN_{pn}(BX, U, V)
$$

where $B$ is $p\times q$ and unknown, $X$ is $(q\times n)$ and known, $U$ is known $(p\times p)$ covariance, and $V$ is $(n\times n)$ known covariance. Suppose $B$ is the prior:

$$
B \sim MN_{pq}(M, U, D)
$$

where $M$ is known $(p\times q)$ mean matrix, $U$ is the restriction mentioned earlier, and $D$ is $(q\times q)$ known covariance. Using Bayes rule, the matrix normal density (which can be found at the respective [Wikipedia page](https://en.wikipedia.org/wiki/Matrix_normal_distribution)), and a lot of algebra (which looks similar to completing the square in the linear regression vector case), we get this conditional distribution result:

$$
B|\Theta \sim MN_{pq}(M^* , U , D^*)
$$

where $D^* = (XV^{-1}X' + D^{-1})^{-1}$ is the second posterior covariance, and 

$$
M^* = (\Theta V^{-1}X' + MD^{-1}) D^*
$$
