from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt

import seaborn as sns

def get_periods(df, column_name):
    df['start'] = df.index
    df['change'] = df[column_name].ne(df[column_name].shift().bfill()).astype(int)
    df_periods = df[df['change'] == 1].copy()
    df_periods['stop'] = df_periods['start'].shift(periods=-1)
    df_periods.dropna(inplace=True)
    return df_periods

def _resample(df, period):
    df = df.copy()

    if period not in ['H', 'D']:
        raise ValueError('Period must be "H" or "D".')

    if 'cycle' in df.columns:
        df['cycle_ok'] = df['cycle'].apply(lambda cycle: 1 if cycle == 'ok' else 0)
        df['cycle_fail'] = df['cycle'].apply(lambda cycle: 1 if cycle == 'fail' else 0)
        df.drop('cycle', axis = 1, inplace = True)

    resampler = df.resample(period)
    
    columns_mean = ['buffer_charge', 'buffer_charge_percentage', 'buffer_voltage', 'sleep']
    if 'd_next' in df.columns:
        columns_mean.append('d_next')
    columns_sum = ['buffer_energy_wasted', 'buffer_failure', 'buffer_in', 'buffer_out', 'consumption', 'cycle_ok', 'cycle_fail', 'solar_intake']

    all_columns = columns_mean + columns_sum + ['timestamp']
    for column in df.columns:
        if column not in all_columns:
            print('Make sure to handle column {}'.format(column))

    df_sum = resampler[columns_sum].sum()
    df_mean = resampler[columns_mean].mean()

    df_x = pd.concat([df_mean, df_sum], sort=False)
    df_x['timestamp'] = df_x.index

    return df_x


def show_solar(df, ax, start, end, device, df_org):
    ax.set_xlim(start, end)
    ax.grid()

    #sns.barplot(df['timestamp'], df['solar_intake'], color='orange', label='Solar Intake [Ws]', ax=ax)
    #sns.barplot(df['timestamp'], df['buffer_energy_wasted'], color='coral', label='Charge Wasted [Ws]', ax=ax)
    
    #ax.fill_between(df['timestamp'].values, 0, df['solar_intake'], hatch='//', color='orange', alpha=0.3, interpolate=True, label='Solar Intake [Ws]')
    #ax.fill_between(df['timestamp'].values, 0, df['buffer_energy_wasted'], hatch='//', color='coral', alpha=0.3, interpolate=True, label='Charge Wasted [Ws]')
    #ax.plot(df['timestamp'], df['solar_intake'], color='orange')
    #ax.plot(df['timestamp'], df['buffer_energy_wasted'], color='coral')

    df['solar_intake_p'] = df['solar_intake'] / df['sleep']
    df['buffer_energy_wasted_p'] = df['buffer_energy_wasted'] / df['sleep']
    ax.fill_between(df['timestamp'], df['solar_intake_p'], step="post", color='orange', alpha=0.3, label='Solar Intake [W]')
    ax.step(df['timestamp'], df['solar_intake_p'], color='orange', where='post')
    ax.fill_between(df['timestamp'], df['buffer_energy_wasted_p'], step="post", color='coral', alpha=0.3, label='Charge Wasted [W]')
    ax.step(df['timestamp'], df['buffer_energy_wasted_p'], color='coral', where='post')

    label = ''
    if 'city' in device.config:
        label = label + device.config['city']
    if 'solar_year' in device.config:
        label = label + ' ({})'.format(device.config['solar_year'])
    ax.text(start + (end-start)/2, df['solar_intake'].min(), label, verticalalignment='bottom', horizontalalignment='center', color='darkgray', fontsize=15)
    ax.legend(loc='upper left')

    if device.solar:
        ax2 = ax.twinx()
        adjusted_timestamps = device.solar.get_timestamps_adjusted_to_year(start.year)
        ax2.step(adjusted_timestamps, device.solar.data['solar'], color='gold', where='post', alpha=0.2)
        ax2.fill_between(adjusted_timestamps, device.solar.data['solar'], step="post", color='gold', alpha=0.2, label='Solar')
        ax2.legend(loc='upper right')


def show_buffer_soc(df, ax, start, end, config, df_org):
    ax.set_xlim(start, end)
    ax.grid()

    #dfp = get_periods(df_org, 'buffer_state')
    #low_power_hysteresis = False
    #for i, row in dfp[dfp['buffer_state']=='low_power_hysteresis'].iterrows():
    #    y = (0,100)
    #    ax.fill_betweenx(y, row['start'], row['stop'], color='orange', hatch='//', alpha=0.2, label='Low Power Hysteresis')
    #    low_power_hysteresis = True
    #if low_power_hysteresis:
    #    ax.axhline(config['low_power_hysteresis'], color='orange', linewidth=2, alpha=0.9, linestyle='--')
    dfp = get_periods(df_org, 'cycle')
    for i, row in dfp[dfp['cycle']=='fail'].iterrows():
        y = (0,100)
        ax.fill_betweenx(y, row['start'], row['stop'], color='red', alpha=0.1)

    ax.plot(df['timestamp'], df['buffer_charge_percentage'], color='black', label='Buffer Charge [%]', linewidth=3)
    # works not on imac:
    #ax.bar(df['timestamp'].values, df['buffer_charge_percentage'], color='black', width=np.timedelta64(24, 'h'))
    #ax.set(ylabel='State of Charge (Percent)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')

def show_buffer_io(df, ax, start, end, config, df_org):
    ax.set_xlim(start, end)
    ax.grid()

    #ax.fill_between(df['timestamp'].values, 0, df['buffer_in'], color='orangered', alpha=0.3, interpolate=True, label='Buffer In [Ws]')
    #ax.fill_between(df['timestamp'].values, 0, - df['buffer_out'], color='crimson', alpha=0.3, interpolate=True, label='Buffer Out [Ws]')
    #ax.plot(df['timestamp'], df['buffer_in'], color='orangered') #, marker='', label='Buffer Intake [Ws]')
    #ax.plot(df['timestamp'], - df['buffer_out'], color='crimson') #, marker='', label='Buffer Outtake [Ws]')

    df['buffer_in_p'] = df['buffer_in'] / df['sleep']
    df['buffer_out_p'] = df['buffer_out'] / df['sleep']
    ax.fill_between(df['timestamp'], df['buffer_in_p'], step="post", color='orangered', alpha=0.3, label='Solar Intake [W]')
    ax.step(df['timestamp'], df['buffer_in_p'], color='orangered', where='post')
    ax.fill_between(df['timestamp'], - df['buffer_out_p'], step="post", color='crimson', alpha=0.3, label='Solar Intake [W]')
    ax.step(df['timestamp'], - df['buffer_out_p'], color='crimson', where='post')

    ax.legend(loc='upper left')


def show_consumption(df, ax, start, end, config, df_org):
    ax.set_xlim(start, end)
    ax.grid()    
    # performance problems:
    #for x in df[df['cycle']=='ok'].timestamp.values:
    #    ax2.axvline(x, color='limegreen',linewidth=1,alpha=0.3)
    #utility = df[df['cycle']=='ok'].timestamp.resample('H').count()
    #ax2.plot(utility.index, utility, color='limegreen')
    
    # where: pre (default), mid, post
    #ax.step(df['timestamp'], df['sleep'], where='post', color='limegreen', label='Sleep time [s]')
    ax.plot(df['timestamp'], df['consumption'], color='royalblue', label='Consumption [W]')
    ax.legend(loc='upper left')


def show_utility(df, ax, start, end, config, df_org):
    ax.set_xlim(start, end)
    ax.grid()    
    # performance problems:
    #for x in df[df['cycle']=='ok'].timestamp.values:
    #    ax2.axvline(x, color='limegreen',linewidth=1,alpha=0.3)
    #utility = df[df['cycle']=='ok'].timestamp.resample('H').count()
    #ax2.plot(utility.index, utility, color='limegreen')
    
    # where: pre (default), mid, post
    #ax.step(df['timestamp'], df['sleep'], where='post', color='limegreen', label='Sleep time [s]')
    #ax.fill_between(df['timestamp'], df['sleep'], step="post", color='limegreen', alpha=0.3, label='Sleep time [s]')
    ax.step(df['timestamp'], df['sleep'], where='post', color='limegreen')
    #ax.step(df['timestamp'], df['duty_cycle'], where='post', color='limegreen', label='Duty Cycle')
    ax.legend(loc='upper left')

def show_duty_cycle(df, ax, start, end, config, df_org):
    ax.set_xlim(start, end)
    ax.grid()    
    ax.step(df['timestamp'], df['duty_cycle'], where='post', color='limegreen', label='Duty Cycle')
    ax.legend(loc='upper left')

def show_device(device, start=None, end=None, resample='auto', extra_plots=None):
    df = device.get_states()
    config = device.config
    
    if start is None:
        start = df['timestamp'].min()
    if end is None:
        end = df['timestamp'].max()

    df['cycle_ok'] = df['cycle'].apply(lambda cycle: 1 if cycle == 'ok' else 0)
    df['cycle_fail'] = df['cycle'].apply(lambda cycle: 1 if cycle == 'fail' else 0)

    df_org = df

    if resample == 'auto':
        duration = end - start
        if (duration > timedelta(days=7)) and (duration <= timedelta(days=100)):
            df = _resample(df, 'H')
        if duration > timedelta(days=100):
            df = _resample(df, 'D')
    elif (resample == 'D') or (resample == 'H'):
        df = _resample(df, 'D')

    functions = []
    if 'solar_intake' in df:
        functions.append(show_solar)
    if 'buffer_charge_percentage' in df:
        functions.append(show_buffer_soc)
    if 'buffer_in' in df:
        functions.append(show_buffer_io)
    if 'consumption' in df:
        functions.append(show_consumption)
    if 'duty_cycle' in df:
        functions.append(show_duty_cycle)
    # uility
    functions.append(show_utility)
    if extra_plots is not None:
        functions.extend(extra_plots)

    #df = df[start:end]
    #fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

    subplots = len(functions)
    fig, axs = plt.subplots(subplots, figsize=(18, 2 * subplots), dpi=200)

    if len(functions) == 0:
        print('Nothing to plot.')
    elif len(functions) == 1:
        functions[0](df, axs, start, end, device, df_org)
    elif len(functions) > 1:
        for i, function in enumerate(functions):
            function(df, axs[i], start, end, device, df_org)

    plt.tight_layout()
    plt.show()