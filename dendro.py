# last update 28/9/2023

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter
import pvlib


class dendro:
    def __init__(self, data, C=0, D=0, period=48, window=2500, latitude=31.3475, longitude=35.0355, time_zone=2):
        self.data = data
        self.name = data.name


        self.d_data = self.data.fillna(method="ffill")
        self.d_data = self.d_data.fillna(method="bfill")

        self.period = period
        self.window = window
        self.latitude = latitude
        self.longitude = longitude
        self.time_zone = time_zone
        self.D = D
        self.C = C

        if D != 0:
            self.C = D*np.pi
        elif C !=0:
            self.D = C/np.pi
        
        self.start = data.first_valid_index()
        self.end = data.last_valid_index()



        # do any additional initialization here
    
# Seasonal decompose 
    def decompose(self, model='additive'):
        trend = getattr(seasonal_decompose(self.d_data, model=model,period=self.period), 'trend')
        return trend

    # rolling average
    def rolling(self, win_type='gaussian'):
        roll = self.d_data.rolling(window=self.window, win_type=win_type, center=True).mean(std=self.window)
        return roll
    

    # create GRO (Accumulated growth assuming zerow growth model)
    def GRO(self, trend=False):
        if trend:
            return self.decompose().expanding().max()
        else:
            return self.d_data.expanding().max()
    
    # Tree water deficit
    def TWD(self, trend=False):
        if trend:
            return self.GRO(trend=True) - self.decompose()
        else:
            return self.GRO() - self.d_data
    
    
    # def TWD_trend(self, model='additive'):
    #     TWD = self.TWD()
    #     trend = getattr(seasonal_decompose(TWD, model=model,period=self.period), 'trend')
    #     return trend
    
    def TWD_rolling(self, win_type='gaussian', ):
        TWD = self.TWD_trend()
        roll = TWD.rolling(window=self.window, win_type=win_type, center=True).mean(std=self.window)
        return roll
    
    def TWD_stationary(self, win_type='gaussian', ):
        roll = self.TWD_rolling()
        TWD_trend = self.TWD_trend()
        return TWD_trend - roll
    
    # daily maximum stem diameter
    def daily_MXSD(self):
        return self.data.resample('D').max()
    
    # daily minimum stem diameter
    def daily_MNSD(self):
        return self.data.resample('D').min()
    
    # maximum daily shrinkage
    def MDS(self):
        return self.daily_MXSD() - self.daily_MNSD()
    
    # daily growth
    def DG(self):
        return self.daily_MXSD.diff(periods=-1)*-1
    
    # daily recovery
    def DR(self):
        return self.daily_MXSD().shift(periods=-1) - self.daily_MNSD()

    # Applying TWD baseling algorithem to the data:
    def TWD_base(self,days_threshold=35):
        data = self.TWD()
        threshold = days_threshold*self.period

        #create list to stor baseline points
        baseline = []

        #add first datapoint (including nan)
        i = 0
        p = data[i]
        p_index = data.index[i]
        baseline.append((p_index, p))

        # jump to first actual data point (not nan)
        i = data.index.get_loc(self.start)
        p = data[i]
        p_index = data.index[i]
        baseline.append((p_index, p))

        # loop for collecting the baseline points

        while i < data.index.get_loc(self.end)-threshold:
            # y and x values of a window after the last baseline point
            y_values = data[i+1:i+threshold+1]
            x_values = np.array(data[i+1:i+threshold+1].index.astype(np.int64) // 10**9)

            # calculate the angles bewteen the last point and every point in the window
            # here we move the data in a way that the last point will be at 0,0 and then we calculate the angle between the vector of the 
            # point in the windo to the vector 1,0
            angles = np.arctan2(y_values - p, x_values - p_index.timestamp())*180 /np.pi

            # if all angles in the window are positive, take the point with the minimum angle as the next baseline point
            if np.all(angles >= 0):
                p_index = angles.idxmin()
                p = data[p_index]

            # if there are negative angles in the window, take the first point with a negative angle as the next baseline point    
            else:
                first_negative_angle_index = angles.lt(0).idxmax()   #lt() does `less than`
                negative_value_at_first_negative_angle_index = data[first_negative_angle_index]
                p = negative_value_at_first_negative_angle_index
                p_index = first_negative_angle_index

            # add point to list and jump to it for the next iteration
            i = i + angles.index.get_loc(p_index) + 1
            baseline.append((p_index, p))
        
        # add the last data point (including nan).
        p_index = data.index[-1]
        baseline.append((p_index, p))

        # create a pandas series of the baseline. this series will have the same index as self.data 
        # and all points between baseline points are filled with interpulated values.
        series = pd.Series(dict(baseline), name=data.name + '_baseline')
        series = series.reindex(data.index)
        series = series.interpolate()

        return series
    
    def TWD_base_rolling(self,days_threshold=20,  window=1250):

        TWD_base = self.TWD_base(days_threshold=days_threshold)
        return TWD_base.rolling(window=window, win_type='gaussian', center=True).mean(std=window)
    
    def TWD_delta(self):
        # TWD_base_rolling = self.TWD_base_rolling()
        
        return self.TWD() - self.TWD_base()
    
    def TWD_delta_trend(self,threshold=20, shift='15T', window=1250, model='additive'):
        TWD_delta = self.TWD_delta()
        trend = getattr(seasonal_decompose(TWD_delta, model=model,period=self.period), 'trend')
        return trend
        # TWD_base = self.TWD_base()
        # # If need to shift the index in 15 min so it would fit with the original index
        # TWD_base.index = TWD_base.index.shift(freq=shift)

        # df = pd.merge(self.d_data, TWD_base, left_index=True, right_index=True, how='outer')
        # df['TWD_baseline'] =  df.iloc[:, -1].interpolate()
        # df['TWD_baseline_rolling'] = df['TWD_baseline'].rolling(window=window, win_type='gaussian', center=True).mean(std=window)
        # TWD_delta = self.TWD() - df['TWD_baseline_rolling']
        # TWD_delta = TWD_delta.fillna(method="ffill")
        # TWD_delta = TWD_delta.fillna(method="bfill")
        # TWD_delta_trend = getattr(seasonal_decompose(TWD_delta, model=model,period=self.period), 'trend')
        # return TWD_delta_trend
    
    def TREX(self, data_is_twd=False):
        if not data_is_twd:
            TWD= self.TWD(trend=True).resample('D').first().fillna(method='ffill')
        else:
            TWD = self.d_data.resample('D').first().fillna(method='ffill')

        TREX = pd.Series([0] * len(TWD), index=TWD.index)
        def calc_TREX(TREX_index,TWD_index):
            if TWD[TREX_index] == 0 or TWD_index < 0:
                return 0
            if TWD[TREX_index] > TWD[TWD_index]:
                return TREX[TWD_index] + 1
            return calc_TREX(TREX_index,TWD_index-1)
        
        for TREX_index in range(1, len(TREX)):
            TREX[TREX_index] = calc_TREX(TREX_index, TREX_index-1)
        
        return TREX

    def TREX_old(self, data_is_twd=False):

        if not data_is_twd:
            TWD_trend = self.TWD(trend=True)*-1
        else:
            TWD_trend = self.d_data*-1

        # Group by day and get the first point for every day
        TWD_trend_daily = TWD_trend.resample('D').first()
        s = TWD_trend_daily
        # Identify where the series decreases
        s_diff = s.diff()
        # Initialize an empty array to store the level
        # level_array = np.zeros(len(s), dtype=int)
        level_array = pd.Series(np.nan, index=s.index)

        # The first entry is always zero
        level_array[0] = 0
        last_level = 0

        # for i, val in enumerate(s):
        for index, val in s.items():
            if s_diff[index] == 0:
                level_array[index] = last_level 
            elif s_diff[index] < 0:
                last_level += 1
                level_array[index] = last_level
            else:
                historical_data = s[:index]

                last_higher_idx = historical_data[historical_data > val].last_valid_index()

                # last_higher_positional_idx = None if last_higher_idx is None else s.index.get_loc(last_higher_idx)
                # print(last_higher_positional_idx)
                if last_higher_idx is None:
                    last_level = 0
                    level_array[index] = last_level
                    continue

                last_level = level_array[last_higher_idx]
                level_array[index] = last_level
        return level_array

    def test(self):
        return 1

    def derivative_df(self, half_period=None):
        if half_period is None:
            half_period = int(self.period /2)
        
        # the reason we are doing this and not d_data is because we want some kind of interpulation for missing data before apllying savgol
        # later on we will remove the big gaps of nan that apear in the original data.
        data_for_sav_gol = self.data.interpolate().fillna(method="ffill").fillna(method="bfill")
        sav_gol = savgol_filter(x=data_for_sav_gol, window_length=half_period, polyorder=2)
        derivative = savgol_filter(x=data_for_sav_gol, window_length=half_period, polyorder=2, deriv=1)

        #create Dataframe
        d_df = pd.DataFrame({'time': self.d_data.index.values,  'derivative': derivative, 'sav_gol': sav_gol, 'data': self.data})
        d_df.set_index('time', inplace=True)
        
        # limit to only relevant time
        d_df = d_df[self.start:self.end]

        d_df.index = d_df.index.floor('T')

        # merge duplicated indexes
        d_df = d_df.groupby(d_df.index).mean()


        # Resample to 1-minute intervals
        d_df = d_df.resample('1T').asfreq()


        # calculate day and night:
        # Calculate solar position
        solar_position = pvlib.solarposition.get_solarposition(
            d_df.index, self.latitude, self.longitude
        )

        solar_position.index = solar_position.index.shift(self.time_zone, freq='H')

        # Add 'day_night' column to df
        # 1 if it's day (elevation > 0), 0 if it's night (elevation <= 0)
        d_df['day_night'] = (solar_position['elevation'] > 0).astype(int)



        # Threshold of nan for interpolation 
        # this section takes care of long periods with nan values that we dont want to include as they will greatly affect the 
        # derivative. so we set a hour limit on the gaps we want to include.
        n_hour_threshold = 3
        thresh_num_of_nan_indecies = n_hour_threshold*60

        # Calculate gap sizes based on non-NaN elements
        gap_sizes = d_df['data'].notna().diff().where(lambda x: x).fillna(0).cumsum()
        gap_sizes = gap_sizes.map(gap_sizes.value_counts())

        # Create a mask where gap sizes are <= thresh_num_of_nan_indecies
        mask = gap_sizes <= thresh_num_of_nan_indecies

        # Interpolate only where the mask is True
        # d_df['data'] = d_df['data'].interpolate(method='index')[mask]

         # Interpolate only where the mask is True
        d_df = d_df.interpolate(method='index')[mask]

 
        return d_df
    

    def lambda_df(self,  window=1, half_period=None):
        if half_period is None:
            half_period = int(self.period /2)

        d_df = self.derivative_df(half_period=half_period)

        df_ratios_by_day = d_df.groupby(pd.Grouper(freq='D'))

        # Count the number of times 'derivative' > 0 for each day
        minutes_expanding_by_day = df_ratios_by_day.apply(lambda x: (x['derivative'] > 0).sum())
        minutes_contracting_by_day = df_ratios_by_day.apply(lambda x: (x['derivative'] < 0).sum())

        # Sum of positive 'derivative' values for each day
        total_expansion_by_day = df_ratios_by_day.apply(lambda x: x[x['derivative'] > 0]['derivative'].sum())
        total_contraction_by_day = df_ratios_by_day.apply(lambda x: x[x['derivative'] < 0]['derivative'].sum())*-1

        if window == 1:
            lambda_t = minutes_contracting_by_day/(minutes_contracting_by_day+minutes_expanding_by_day)
            lambda_a = total_contraction_by_day/(total_contraction_by_day+total_expansion_by_day)

            l_df = pd.DataFrame({'time': minutes_expanding_by_day.index.values,  'lambda_t': lambda_t, 'lambda_a': lambda_a})
            l_df.set_index('time', inplace=True)

            return l_df
        
        else:
            # # Create a list to store the date groups
            l_df = pd.DataFrame({'time': minutes_expanding_by_day.index.values,  'lambda_t': np.nan, 'lambda_a': np.nan})
            l_df.set_index('time', inplace=True)
            # Iterate through the DataFrame to create groups
            for i in range(window, len(df_ratios_by_day)):
                sum_minutes_expanding_by_day = minutes_expanding_by_day[i-window:i].sum()
                sum_minutes_contracting_by_day = minutes_contracting_by_day[i-window:i].sum()

                sum_total_expansion_by_day = total_expansion_by_day[i-window:i].sum()
                sum_total_contraction_by_day = total_contraction_by_day[i-window:i].sum()
                
                lambda_t = sum_minutes_contracting_by_day/(sum_minutes_contracting_by_day+sum_minutes_expanding_by_day)
                lambda_a = sum_total_contraction_by_day/(sum_total_contraction_by_day+sum_total_expansion_by_day)

                # l_df.at[i, 'lambda_t'] = lambda_t
                l_df.iloc[i, l_df.columns.get_loc('lambda_t')] = lambda_t
                # l_df.at[i, 'lambda_a'] = lambda_a
                l_df.iloc[i, l_df.columns.get_loc('lambda_a')] = lambda_a
            
            return l_df
        



    def lambda_df_day_night(self,  window=1, half_period=None):
        if half_period is None:
            half_period = int(self.period /2)

        d_df = self.derivative_df(half_period=half_period)

        day_data = d_df.loc[d_df['day_night'] == 1]
        night_data = d_df.loc[d_df['day_night'] == 0]

        daily_night_data = night_data.groupby(pd.Grouper(freq='D'))
        daily_day_data = day_data.groupby(pd.Grouper(freq='D'))
        daily_all_data = d_df.groupby(pd.Grouper(freq='D'))

        day_to_night_ratio_single = daily_all_data['day_night'].mean()

        datas = [daily_night_data, daily_day_data, daily_all_data]

        minutes_expanding = []
        minutes_contracting = []
        total_expansion = []
        total_contraction = []
        absolute_area = []

        lamda_ts = []
        lamda_as = []

        for data in datas:
            # Count the number of times 'derivative' > 0 for each day
            m_ex = data.apply(lambda x: (x['derivative'] > 0).sum())
            m_cont = data.apply(lambda x: (x['derivative'] < 0).sum())

            # Sum of positive or negative 'derivative' values for each day (Area = integral)
            t_ex = data.apply(lambda x: x[x['derivative'] > 0]['derivative'].sum())
            t_cont = data.apply(lambda x: x[x['derivative'] < 0]['derivative'].sum())*-1

            # absolute_area Sum of absolute 'derivative' values for each day (Area_absolute = integral)
            a_a = data.apply(lambda x: x['derivative'].abs().sum())

            minutes_expanding.append(m_ex)
            minutes_contracting.append(m_cont)
            total_expansion.append(t_ex)
            total_contraction.append(t_cont)
            absolute_area.append(a_a)

        if window == 1:

            for i in range(3):

                lambda_t = minutes_contracting[i]/(minutes_contracting[i]+minutes_expanding[i])
                lambda_a = total_contraction[i]/(total_contraction[i]+total_expansion[i])

                lamda_ts.append(lambda_t)
                lamda_as.append(lambda_a)
            
            lambda_DNAa = absolute_area[1]/(absolute_area[1]+absolute_area[0])

            l_df = pd.DataFrame({'time': minutes_expanding[2].index.values, 
                                  'lambda_t': lamda_ts[2], 'lambda_a': lamda_as[2],
                                  'lambda_t_night': lamda_ts[0], 'lambda_a_night': lamda_as[0],
                                  'lambda_t_day': lamda_ts[1], 'lambda_a_day': lamda_as[1],
                                  'day_to_night_ratio_single': day_to_night_ratio_single,
                                  'lambda_DNt': day_to_night_ratio_single,
                                  'lambda_DNAa': lambda_DNAa})
            l_df.set_index('time', inplace=True)

            return l_df
        
        else:
            # # Create a list to store the date groups
            l_df = pd.DataFrame({'time': minutes_expanding[2].index.values, 
                                'lambda_t': np.nan, 'lambda_a': np.nan,
                                'lambda_t_night': np.nan, 'lambda_a_night': np.nan,
                                'lambda_t_day': np.nan, 'lambda_a_day': np.nan, 
                                'day_to_night_ratio_single': day_to_night_ratio_single,
                                'lambda_DNt': np.nan,
                                'lambda_DNAa': np.nan})
            l_df.set_index('time', inplace=True)

            # Iterate through the DataFrame to create groups
            for j in range(3):
                for i in range(window, min(len(daily_night_data), len(daily_day_data), len(daily_all_data))):

                    sum_minutes_expanding = minutes_expanding[j][i-window:i].sum()
                    sum_minutes_contracting = minutes_contracting[j][i-window:i].sum()

                    sum_total_expansion = total_expansion[j][i-window:i].sum()
                    sum_total_contraction = total_contraction[j][i-window:i].sum()

                    lambda_t = sum_minutes_contracting/(sum_minutes_contracting+sum_minutes_expanding)
                    lambda_a = sum_total_contraction/(sum_total_contraction+sum_total_expansion)

                    if j == 2:
                        l_df.iloc[i, l_df.columns.get_loc('lambda_t')] = lambda_t
                        l_df.iloc[i, l_df.columns.get_loc('lambda_a')] = lambda_a

                        # calculatng absolute area
                        sum_absolute_area_day = absolute_area[1][i-window:i].sum()
                        sum_absolute_area_night = absolute_area[0][i-window:i].sum()

                        lambda_DNAa = sum_absolute_area_day/(sum_absolute_area_day+sum_absolute_area_night)
                        l_df.iloc[i, l_df.columns.get_loc('lambda_DNAa')] = lambda_DNAa

                        #calculating lambda_DNt for the window
                        lambda_DNt = day_to_night_ratio_single[i-window:i].mean()
                        l_df.iloc[i, l_df.columns.get_loc('lambda_DNt')] = lambda_DNt

                    elif j == 0:
                        l_df.iloc[i, l_df.columns.get_loc('lambda_t_night')] = lambda_t
                        l_df.iloc[i, l_df.columns.get_loc('lambda_a_night')] = lambda_a
                    elif j == 1:
                        l_df.iloc[i, l_df.columns.get_loc('lambda_t_day')] = lambda_t
                        l_df.iloc[i, l_df.columns.get_loc('lambda_a_day')] = lambda_a
            
            return l_df

     