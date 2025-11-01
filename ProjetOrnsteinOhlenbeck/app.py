
from flask import Flask, render_template, request
import yfinance as yf
from Tools import cov, corr, moy, std
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime,timedelta
from fractions import Fraction

app = Flask(__name__)

# Télécharger les données pour l'arbitrage
symbols = ["AAPL", "MSFT"]
data = yf.download(symbols, period="6mo", interval="1h")
microsoft = data['Open']['MSFT'].to_numpy()
apple = data['Open']['AAPL'].to_numpy()
timestamps = data.index  # Récupérer les timestamps des données


# Fonction de calcul des résultats
def model_spread(n, t, nbsim):
    correlation = corr(apple, microsoft)
    spread = microsoft - apple
    spread = spread[~np.isnan(spread)]
    spread0 = spread[-1] #a[-1]	Accède au dern élément.a[:-1]Accède à tous les éléments sauf le dern.

    # Régression pour estimer theta
    delta_spread = spread[1:] - spread[:-1]
    spread_lagged = spread[:-1]
    X = -spread_lagged.reshape(-1, 1) #pour obtenir un tableau 2d pour la regression lineaire
    y = delta_spread
    reg = LinearRegression().fit(X, y) #y=theta*X+c
    theta = reg.coef_[0]

    m = moy(spread)
    sigma = std(spread)

    # Modélisation du spread (Ornstein-Uhlenbeck)
    z = np.random.randn(nbsim, n - 1)
    spreadmod = np.zeros((nbsim, n))
    spreadmod[:, 0] = spread0

    for j in range(nbsim):
        for i in range(1, n):
            spreadmod[j, i] = spreadmod[j, i - 1] + theta * (m - spreadmod[j, i - 1]) * t + sigma * np.sqrt(t) * z[
                j, i - 1]

    spreadmoy = np.mean(spreadmod, axis=0)
    return spreadmod, spreadmoy, theta, m, sigma, correlation



import plotly.graph_objects as go
import plotly.io as pio

def generate_interactive_plots(spreadmod, spreadmoy, timestamps):
    """
    Génère deux graphiques interactifs en Plotly :
      - Un pour toutes les simulations du spread.
      - Un pour le spread moyen.

    :param spreadmod: tableau numpy des simulations du spread (dimensions : nbsim x n)
    :param spreadmoy: tableau numpy du spread moyen (longueur n)
    :param timestamps: liste de timestamps (chaînes) de longueur n
    :return: graph1_html, graph2_html (chaînes HTML à insérer dans le template)
    """
    # Graphique interactif pour les simulations
    fig1 = go.Figure()
    nbsim = spreadmod.shape[0]
    for j in range(nbsim):
        fig1.add_trace(go.Scatter(
            x=timestamps,
            y=spreadmod[j],
            mode='lines',
            name=f'Simulation {j + 1}',
            line=dict(width=0.8)
        ))
    fig1.update_layout(
        title="Simulations du Spread selon le Processus d'Ornstein-Uhlenbeck",
        xaxis_title="Temps (jours et heures)",
        yaxis_title="Spread",
        xaxis=dict(tickangle=45)
    )
    graph1_html = pio.to_html(fig1, full_html=False)

    # Graphique interactif pour le spread moyen
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=timestamps,
        y=spreadmoy,
        mode='lines',
        name='Spread moyen',
        line=dict(color='blue', width=2)
    ))
    fig2.update_layout(
        title="Spread moyen selon le Processus d'Ornstein-Uhlenbeck",
        xaxis_title="Temps (jours et heures)",
        yaxis_title="Spread moyen",
        xaxis=dict(tickangle=45)
    )
    graph2_html = pio.to_html(fig2, full_html=False)

    return graph1_html, graph2_html


@app.route('/', methods=['GET', 'POST'])
def index():
    n = 20
    t = 1 / (252 * 6)
    nbsim = 100
    period = "6mo"
    interval = "1h"

    if request.method == 'POST':
        period = request.form['period']
        interval = request.form['interval']
        n = int(request.form['n'])
        t = request.form['t']
        if '/' in t:
            t = float(Fraction(t))  # Convertit "1/252" en 0.00397...
        else:
            t = float(t)  # Convertit normalement
        nbsim = int(request.form['nbsim'])

    spreadmod, spreadmoy, theta, m, sigma, correlation = model_spread(n, t, nbsim)

    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    # Définition des seuils
    seuil_sup = np.mean(spreadmoy) + np.std(spreadmoy)
    seuil_inf = np.mean(spreadmoy) - np.std(spreadmoy)

    # Déterminer l'incrément de temps en fonction de l'intervalle
    def parse_interval(interval):
        if interval.endswith('h'):
            return timedelta(hours=int(interval[:-1]))
        elif interval.endswith('d'):
            return timedelta(days=int(interval[:-1]))
        else:
            raise ValueError("Intervalle non reconnu. Utilisez '1h' pour une heure ou '1d' pour un jour.")

    interval_td = parse_interval(interval)

    # Ajustement des horaires de marché
    def adjust_start_time(start_time):
        """ Ajuste start_time pour qu'il tombe sur un jour ouvré à 9h30 """
        market_open = 9 * 60 + 30
        market_close = 16 * 60

        # Si c'est un week-end, avancer jusqu'au lundi
        if start_time.weekday() >= 5:
            start_time += timedelta(days=(7 - start_time.weekday()))
            start_time = start_time.replace(hour=9, minute=30)

        # Si c'est avant l'ouverture, ajuster à 9h30
        current_minutes = start_time.hour * 60 + start_time.minute
        if current_minutes < market_open:
            start_time = start_time.replace(hour=9, minute=30)
        # Si c'est après la fermeture, avancer au jour suivant à 9h30
        elif current_minutes >= market_close:
            start_time += timedelta(days=1)
            if start_time.weekday() >= 5:
                start_time += timedelta(days=(7 - start_time.weekday()))
            start_time = start_time.replace(hour=9, minute=30)

        return start_time

    # Initialisation
    now = datetime.now()
    now_adjusted = adjust_start_time(now)
    trade_time = now_adjusted

    trades = []
    position = None
    entry_price = None
    entry_time = None
    profits = []

    # Stratégie d'arbitrage
    for s in spreadmoy:
        trade_time = adjust_start_time(trade_time)  # Ajustement avant d'utiliser l'heure

        if position is None:
            if s > seuil_sup:
                position = 'short'
                entry_price = s
                entry_time = trade_time  # Sauvegarde du bon timestamp
                trades.append({
                    'type': 'short',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': s,
                    'exit_time': None,
                    'exit_price': None,
                    'profit': None
                })
            elif s < seuil_inf:
                position = 'long'
                entry_price = s
                entry_time = trade_time  # Sauvegarde du bon timestamp
                trades.append({
                    'type': 'long',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': s,
                    'exit_time': None,
                    'exit_price': None,
                    'profit': None
                })
        elif position == 'short' and s <= np.mean(spreadmoy):
            profit = entry_price - s
            profits.append(profit)
            trades[-1]['exit_time'] = trade_time.strftime('%Y-%m-%d %H:%M')  # Correct timestamp
            trades[-1]['exit_price'] = s
            trades[-1]['profit'] = profit
            position = None
        elif position == 'long' and s >= np.mean(spreadmoy):
            profit = s - entry_price
            profits.append(profit)
            trades[-1]['exit_time'] = trade_time.strftime('%Y-%m-%d %H:%M')  # Correct timestamp
            trades[-1]['exit_price'] = s
            trades[-1]['profit'] = profit
            position = None

        trade_time += interval_td  # Mise à jour de trade_time après le traitement

    # Convertir en DataFrame pour une analyse plus facile
    trades_df = pd.DataFrame(trades)
    trades_html =trades_df.to_html(classes="table table-striped", index=False)
    # Calcul des métriques de performance
    total_profit = sum(profits)
    num_trades = len(trades_df)
    winning_trades = trades_df[trades_df['profit'] > 0]
    win_rate = len(winning_trades) / num_trades
    cumulative_profits = np.cumsum(profits)
    max_drawdown = np.max(np.maximum.accumulate(cumulative_profits) - cumulative_profits)
    risk_free_rate = 0.01  # Taux sans risque annuel
    sharpe_ratio = (np.mean(profits) - risk_free_rate) / np.std(profits)



    # Générer les graphiques interactifs
    plot1, plot2= generate_interactive_plots(spreadmod, spreadmoy, timestamps)




    return render_template('index.html',
                           theta=theta, mu=m, sigma=sigma, correlation=correlation,plot1=plot1,plot2=plot2,
                           period=period, interval=interval, n=n, t=t, nbsim=nbsim,
                           seuil_sup=seuil_sup, seuil_inf=seuil_inf, trades_html=trades_html, total_profit=total_profit,
                           win_rate=win_rate, max_drawdown=max_drawdown, sharpe_ratio=sharpe_ratio)


if __name__ == '__main__':
    app.run(debug=True)


