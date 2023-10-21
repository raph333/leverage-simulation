# Leverage simulation

### Method:
* Data: daily historical prices of the SP500 index starting in 1927 obtained from yahoo finance
* The leverage is applied daily like it would with a leveraged ETF
* For each simulation-run, a timeframe of `n_years` consecutive years is randomly selected
* For each run and for each leverage-value, the multiple of the return is calculated
* This is repeated `n_simulations` times
* The distribution of outcomes is analyzed

### Purpose:
* Just out of curiosity estimating the effect of leverage on a pension portfolio (see [this book](https://www.lifecycleinvesting.net/book.html)). (Needless to say, this is not investment advice.)
* Testing ChatGPT to code faster: it's quite useful for creating plots and refactoring