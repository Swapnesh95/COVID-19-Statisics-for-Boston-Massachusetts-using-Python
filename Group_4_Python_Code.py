#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
%matplotlib inline

def plot_df(df, x, y, title="", xlabel='Date', ylabel='Number of Cases', dpi=250):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


plot_df(data, x=data['Date'], y=data['Positive New'], title='Number of New COVID-19 Cases From 01/01/2022 To 10/16/2022')

def plot_df(df, x, y, title="", xlabel='Date', ylabel='Number of Cases', dpi=250):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


plot_df(data, x=data['Date'], y=data['Confirmed Deaths'], title='Number of COVID-19 Deaths From 01/01/2022 To 10/16/2022')

#Diving The Dataset By Age
age_19 = data.loc[data['Age'] == '0-19']
age_29 = data.loc[data['Age'] == '20-29']
age_39 = data.loc[data['Age'] == '30-39']
age_49 = data.loc[data['Age'] == '40-49']
age_59 = data.loc[data['Age'] == '50-59']
age_69 = data.loc[data['Age'] == '60-69']
age_79 = data.loc[data['Age'] == '70-79']
age_80 = data.loc[data['Age'] == '80+']
age_un = data.loc[data['Age'] == 'Unknown']


age_19.describe()
age_29.describe()
age_39.describe()
age_49.describe()
age_59.describe()
age_69.describe()
age_79.describe()
age_80.describe()
age_un.describe()

data_c = pd.read_csv('DSET_County.csv')

#Creating Dummy Variables
county_Barnstable = data_c.loc[data_c['County'] == 'Barnstable']
county_Berkshire = data_c.loc[data_c['County'] == 'Berkshire']
county_Bristol = data_c.loc[data_c['County'] == 'Bristol']
county_Dukes = data_c.loc[data_c['County'] == 'Dukes']
county_Essex = data_c.loc[data_c['County'] == 'Essex']
county_Franklin = data_c.loc[data_c['County'] == 'Franklin']
county_Hampden = data_c.loc[data_c['County'] == 'Hampden']
county_Hampshire = data_c.loc[data_c['County'] == 'Hampshire']
county_Middlesex = data_c.loc[data_c['County'] == 'Middlesex']
county_Nantucket = data_c.loc[data_c['County'] == 'Nantucket']
county_Norfolk = data_c.loc[data_c['County'] == 'Norfolk']
county_Plymouth = data_c.loc[data_c['County'] == 'Plymouth']
county_Suffolk = data_c.loc[data_c['County'] == 'Suffolk']
county_Worcester = data_c.loc[data_c['County'] == 'Worcester']
county_Unknown = data_c.loc[data_c['County'] == 'Unknown']
county_Dukes = data_c.loc[data_c['County'] == 'Dukes and Nantucket']

def plot_df(df, x, y,a,b, title="", xlabel='Date', ylabel='Number of Cases', dpi=250):
    plt.figure(figsize=(25,4), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.plot(a, b, color='tab:orange')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation = 90)
    plt.show()


plot_df(data, x=age_29['Date'], y=age_29['Cases_Last2Weeks'], title='Number of New COVID-19 Cases From 01/01/2022 To 10/16/2022',a=age_39['Date'], b=age_39['Cases_Last2Weeks'])


class BarebonesSEIR:
    def __init__(self, params=None):
        self.params = params

    def get_fit_params(self):
        params = lmfit.Parameters()
        params.add("population", value=12_000_000, vary=False)
        params.add("epidemic_started_days_ago", value=10, vary=False)
        params.add("r0", value=4, min=3, max=5, vary=True)
        params.add("alpha", value=0.0064, min=0.005, max=0.0078, vary=True)  # CFR
        params.add("delta", value=1/3, min=1/14, max=1/2, vary=True)  # E -> I rate
        params.add("gamma", value=1/9, min=1/14, max=1/7, vary=True)  # I -> R rate
        params.add("rho", expr='gamma', vary=False)  # I -> D rate
        return params

    def get_initial_conditions(self, data):
        # Simulate such initial params as to obtain as many deaths as in data
        population = self.params['population']
        epidemic_started_days_ago = self.params['epidemic_started_days_ago']

        t = np.arange(epidemic_started_days_ago)
        (S, E, I, R, D) = self.predict(t, (population - 1, 0, 1, 0, 0))

        I0 = I[-1]
        E0 = E[-1]
        Rec0 = R[-1]
        D0 = D[-1]
        S0 = S[-1]
        return (S0, E0, I0, Rec0, D0)

    def step(self, initial_conditions, t):
        population = self.params['population']
        delta = self.params['delta']
        gamma = self.params['gamma']
        alpha = self.params['alpha']
        rho = self.params['rho']
        
        rt = self.params['r0'].value
        beta = rt * gamma

        S, E, I, R, D = initial_conditions

        new_exposed = beta * I * (S / population)
        new_infected = delta * E
        new_dead = alpha * rho * I
        new_recovered = gamma * (1 - alpha) * I

        dSdt = -new_exposed
        dEdt = new_exposed - new_infected
        dIdt = new_infected - new_recovered - new_dead
        dRdt = new_recovered
        dDdt = new_dead

        assert S + E + I + R + D - population <= 1e10
        assert dSdt + dIdt + dEdt + dRdt + dDdt <= 1e10
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def predict(self, t_range, initial_conditions):
        ret = odeint(self.step, initial_conditions, t_range)
        return ret.T


import lmfit
from scipy.integrate import odeint

train_subset = age_29[(age_29.Date <= '9/13/2020')]


model = BarebonesSEIR()
model.params = model.get_fit_params()
train_initial_conditions = model.get_initial_conditions(train_subset)
train_t = np.arange(len(train_subset))
(S, E, I, R, D) = model.predict(train_t, train_initial_conditions)
plt.figure(figsize=(15, 7),dpi=250)
plt.plot(train_subset.Date, train_subset['Cases_Last2Weeks'], label='ground truth')
plt.plot(train_subset.Date, D, label='predicted', color='black', linestyle='dashed' )
plt.legend()
plt.xticks(rotation=90)
plt.title('Total deaths')
plt.show()

def sigmoid(x, xmin, xmax, a, b, c, r):
    x_scaled = (x - xmin) / (xmax - xmin)
    out = (a * np.exp(c * r) + b * np.exp(r * x_scaled)) / (np.exp(c * r) + np.exp(x_scaled * r))
    return out


def stepwise_soft(t, coefficients, r=20, c=0.5):
    t_arr = np.array(list(coefficients.keys()))

    min_index = np.min(t_arr)
    max_index = np.max(t_arr)

    if t <= min_index:
        return coefficients[min_index]
    elif t >= max_index:
        return coefficients[max_index]
    else:
        index = np.min(t_arr[t_arr >= t])

    if len(t_arr[t_arr < index]) == 0:
        return coefficients[index]
    prev_index = np.max(t_arr[t_arr < index])
    # sigmoid smoothing
    q0, q1 = coefficients[prev_index], coefficients[index]
    out = sigmoid(t, prev_index, index, q0, q1, c, r)
    return out

t_range = np.arange(100)
coefficients = {
    0: 0,
    30: 0.5,
    60: 1,
    100: 0.4,
}

plt.title('Quarantine function example')
plt.scatter(coefficients.keys(), coefficients.values(), label='Points of change of quarantine measures')
plt.plot(t_range, [stepwise_soft(t, coefficients, r=20, c=0.5) for t in t_range], label='Smooth stepwise function')
plt.xlabel('t')
plt.ylabel('Qaurantine level')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6),)
plt.show()


from sir_models.fitters import HiddenCurveFitter
from sir_models.models import SEIRHidden
stepwize_size = 60
weights = {
    'I': 0.25,
    'R': 0.25,
    'D': 0.5,
}
model = SEIRHidden(stepwise_size=stepwize_size)
fitter = HiddenCurveFitter(
     new_deaths_col='deaths_per_day_ma7',
     new_cases_col='infected_per_day_ma7',
     new_recoveries_col='recovered_per_day_ma7',
     weights=weights,
     max_iters=1000,
)
fitter.fit(model, train_subset)
result = fitter.result