import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definizione dei simboli delle azioni da considerare nel portafoglio
symbols = ['GOOG', 'AMZN', 'META', 'MSFT']

# Ottenimento dei dati storici delle azioni utilizzando yfinance
data = yf.download(symbols, start='2015-01-01', end='2023-01-01')['Adj Close']

# Calcolo dei rendimenti giornalieri logaritmici
daily_returns = np.log(data / data.shift(1))

# Calcolo delle medie dei rendimenti e della matrice di covarianza
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# Definizione del numero di simulazioni Monte Carlo
num_simulations = 10000

# Inizializzazione di array per memorizzare risultati delle simulazioni
num_assets = len(symbols)
results = np.zeros((num_assets + 3, num_simulations))  # Aggiunto +3 per le colonne Return, Risk, Sharpe
risk_free_rate = 0.01  # Tasso di rendimento privo di rischio

# Simulazione dei portafogli
for i in range(num_simulations):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

    results[0,i] = portfolio_return
    results[1,i] = portfolio_stddev
    results[2,i] = sharpe_ratio

    for j in range(len(weights)):
        results[j+3,i] = weights[j]  # Ora gli indici partono da 3

# Creazione del DataFrame dei risultati
columns = ['Return', 'Risk', 'Sharpe'] + symbols
results_df = pd.DataFrame(results.T, columns=columns)

# Identificazione del portafoglio con il massimo Sharpe ratio
max_sharpe_portfolio = results_df.iloc[results_df['Sharpe'].idxmax()]

# Identificazione del portafoglio con il minimo rischio
min_risk_portfolio = results_df.iloc[results_df['Risk'].idxmin()]

# Stampa dei risultati
print("Portafoglio con il massimo Sharpe ratio:")
print(max_sharpe_portfolio)

print("\nPortafoglio con il minimo rischio:")
print(min_risk_portfolio)

# Visualizzazione del trade-off rischio-rendimento
plt.scatter(results_df.Risk, results_df.Return, c=results_df.Sharpe, cmap='YlGnBu', marker='o')
plt.title('Rischio vs Rendimento')
plt.xlabel('Rischio (Deviazione Standard)')
plt.ylabel('Rendimento atteso')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_portfolio.Risk, max_sharpe_portfolio.Return, color='r', marker='*', s=100, label='Massimo Sharpe Ratio')
plt.scatter(min_risk_portfolio.Risk, min_risk_portfolio.Return, color='g', marker='*', s=100, label='Minimo Rischio')
plt.legend()
plt.show()
