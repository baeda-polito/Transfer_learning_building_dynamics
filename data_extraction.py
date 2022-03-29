import pandas as pd
import numpy as np
import h5py
import os
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import csv
from csv import DictWriter
import openpyxl
import xlsxwriter

# Converting factors
J_to_kWh = 1/3600000
area = 4982.22 # m2
kBtu_to_kWh = 0.293071
ft2_to_m2 = 0.092903
kBtu_per_ft2_to_kWh_per_m2 = kBtu_to_kWh/ft2_to_m2

# ____________________________ACCESS THE DATA_____________________________________
dir_to_HDF = 'E:/tesi_messina/A_Synthetic_Building_Operation_Dataset.h5'
hdf = h5py.File(dir_to_HDF, 'r')

# Utility functions
def get_df_from_hdf(hdf, climate, efficiency, year, str_run, data_key):
    '''
    This function extracts the dataframe from the all run hdf5 file
    '''
    ts_root = hdf.get('3. Data').get('3.2. Timeseries')
    sub = ts_root.get(climate).get(efficiency).get(year).get(str_run).get(data_key)
    cols = np.array(sub.get('axis0'))[1:].astype(str)
    data = np.array(sub.get('block1_values'))
    df = pd.DataFrame(data, columns=cols)
    # end_year = str(int(year)+1)
    #df.index = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in pd.date_range(year+'-01-01', year+'-12-31', freq='10min')]#[:-1]
    return df


cols = ['AirSystemOutdoorAirEconomizerStatus', 'CoolingElectricity', 'ElectricityFacility', 'ElectricityHVAC',
            'ExteriorLightsElectricity','FanAirMassFlowRate', 'FanElectricPower',
            'FansElectricity', 'GasFacility', 'GasHVAC', 'HeatingElectricity',
            'InteriorEquipmentElectricity', 'InteriorLightsElectricity',
            'PumpElectricPower', 'PumpMassFlowRate', 'PumpsElectricity',
            'SiteDayTypeIndex', 'SiteHorizontalInfraredRadiationRateperArea',
            'SiteOutdoorAirDewpointTemperature', 'SiteOutdoorAirDrybulbTemperature',
            'SiteOutdoorAirRelativeHumidity', 'SiteOutdoorAirWetbulbTemperature',
            'SystemNodeMassFlowRate', 'SystemNodePressure', 'SystemNodeRelativeHumidity',
            'SystemNodeTemperature', 'ZoneAirRelativeHumidity', 'ZoneAirTerminalVAVDamperPosition',
            'ZoneElectricEquipmentElectricPower', 'ZoneLightsElectricPower', 'ZoneMeanAirTemperature',
            'ZoneMechanicalVentilationMassFlowRate', 'ZonePeopleOccupantCount',
            'ZoneThermostatCoolingSetpointTemperature', 'ZoneThermostatHeatingSetpointTemperature']


cols_list = {
    'AirSystemOutdoorAirEconomizerStatus': [],
    'CoolingElectricity': [],
    'ElectricityFacility': [],
    'ElectricityHVAC': [],
    'ExteriorLightsElectricity': [],
    'FanAirMassFlowRate': [],
    'FanElectricPower': [],
    'FansElectricity': [],
    'GasFacility': [],
    'GasHVAC': [],
    'HeatingElectricity': [],
    'InteriorEquipmentElectricity': [],
    'InteriorLightsElectricity': [],
    'PumpElectricPower': [],
    'PumpMassFlowRate': [],
    'PumpsElectricity': [],
    'SiteDayTypeIndex': [],
    'SiteHorizontalInfraredRadiationRateperArea': [],
    'SiteOutdoorAirDewpointTemperature': [],
    'SiteOutdoorAirDrybulbTemperature': [],
    'SiteOutdoorAirRelativeHumidity': [],
    'SiteOutdoorAirWetbulbTemperature': [],
    'SystemNodeMassFlowRate': [],
    'SystemNodePressure': [],
    'SystemNodeRelativeHumidity': [],
    'SystemNodeTemperature': [],
    'ZoneAirRelativeHumidity': [],
    'ZoneAirTerminalVAVDamperPosition': [],
    'ZoneElectricEquipmentElectricPower': [],
    'ZoneLightsElectricPower': [],
    'ZoneMeanAirTemperature': [],
    'ZoneMechanicalVentilationMassFlowRate': [],
    'ZonePeopleOccupantCount': [],
    'ZoneThermostatCoolingSetpointTemperature': [],
    'ZoneThermostatHeatingSetpointTemperature': []
    }



# ______________________________________CREATE A DATAFRAME WITH ALL COLUMNS NAMES_______________________________________
"""
new_list = cols_list
def list_dictionary(list_name, climate, efficiency, year, str_run):
    for name, values in list_name.items():
        df = get_df_from_hdf(hdf, climate=climate, efficiency=efficiency, year=year, str_run=str_run, data_key=name)
        list_name[name] = list(df.columns)
    return list_name

# sf_list = list_dictionary(new_list, '3C', 'Standard', 'TMY3', 'run_2') # San Francisco
# miami_list = list_dictionary(cols_list, '1A', 'Standard', 'TMY3', 'run_1') # Miami
# sf_list == miami_list # (True)

# ______________________________________SAVE THE LIST IN A EXCEL FORMAT_________________________________________________
def save_cols_name(list, file_name):
    # df = pd.DataFrame([(k, pd.Series(v)) for k, v in cols_list.items()])
    # df.to_excel('list.xlsx', header=False, index=False)

    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()

    col_num = 0
    for key, value in list.items():
        worksheet.write(0, col_num, key)
        worksheet.write_column(1, col_num, value)
        col_num += 1
    workbook.close()


#save_cols_name(cols_list, 'cols_list_Miami.xlsx')
#save_cols_name(new_list, 'cols_list_San_Francisco.xlsx')


resultFile = open('df_sf_cols.csv', 'w')
for name in df_sf.columns():
        resultFile.write(name + "\n")

resultFile.close()
"""

# ____________________________________________________GET TIME SERIES DATA______________________________________________
def get_ts_data(hdf, str_clm, str_eff, str_yr, str_run):
    '''
    This function extracts the time series data from the HDF5 file
    '''
    ts_root = hdf.get('3. Data').get('3.2. Timeseries')
    ts_root.get(str_clm).get(str_eff).get(str_yr).get(str_run).keys()
    df_oat = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'SiteOutdoorAirDrybulbTemperature')
    df_ele = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ElectricityFacility')
    df_mels = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'InteriorEquipmentElectricity')
    df_lights = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'InteriorLightsElectricity')
    df_occs = pd.DataFrame(
        get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ZonePeopleOccupantCount').sum(axis=1),
        columns=['Total Occupant Count'])
    df_havc_ele = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ElectricityHVAC')
    df_gas = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'GasFacility')
    df_HVAC_cooling = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'CoolingElectricity')
    df_HVAC_heating = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'HeatingElectricity')
    df_fan_elec = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'FansElectricity')
    df_pumps_elec = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'PumpsElectricity')
    df_rad = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'SiteHorizontalInfraredRadiationRateperArea')

    df_col = pd.concat([df_occs, df_lights, df_mels, df_ele, df_gas, df_havc_ele, df_oat, df_HVAC_cooling, df_HVAC_heating, df_fan_elec, df_pumps_elec, df_rad], axis=1)
    # df_col['datetime'] = pd.to_datetime(df_col.index)
    df_col['climate'] = str_clm
    df_col['efficiency'] = str_eff
    # df_col['weekday'] = np.where(df_col['datetime'].dt.dayofweek < 5, True, False)
    df_col['Interior Lighting (kWh)'] = df_col['InteriorLights:Electricity[J]'] * J_to_kWh
    df_col['MELs (kWh)'] = df_col['InteriorEquipment:Electricity[J]'] * J_to_kWh
    df_col['Site Electricity (kWh)'] = df_col['Electricity:Facility[J]'] * J_to_kWh
    df_col['Site Gas (kWh)'] = df_col['Gas:Facility[J]'] * J_to_kWh
    df_col['Site Total Energy (kWh)'] = df_col['Site Electricity (kWh)'] + df_col['Site Gas (kWh)'] #somma i KWh di elettricità e gas
    df_col['Cooling electricity (KWh)'] = df_col['Cooling:Electricity[J]'] * J_to_kWh
    df_col['Heating electricity (KWh)'] = df_col['Heating:Electricity[J]'] * J_to_kWh
    df_col['Fans Electricity (KWh)'] = df_col['Fans:Electricity[J]']*J_to_kWh
    df_col['Pumps Electricity (KWh)'] = df_col['Pumps:Electricity[J]'] * J_to_kWh
    df_col['HVAC Electricity (kWh)'] = df_col['Electricity:HVAC[J]'] * J_to_kWh
    df_col['Somma electricity (KWh)'] = df_col['Cooling electricity (KWh)'] + df_col['Heating electricity (KWh)'] + df_col['Fans Electricity (KWh)'] + df_col['Pumps Electricity (KWh)']
    df_col['Outdoor Air Temperature (degC)'] = df_col['Environment:Site Outdoor Air Drybulb Temperature[C]']
    df_col['Horizontal Radiation per Area [W/m2]'] = df_col['Environment:Site Horizontal Infrared Radiation Rate per Area[W/m2]']

    df_col = df_col.drop(['InteriorLights:Electricity[J]',
                          'InteriorEquipment:Electricity[J]',
                          'Electricity:Facility[J]',
                          'Gas:Facility[J]',
                          'Environment:Site Outdoor Air Drybulb Temperature[C]'
                          ], axis=1)
    #df_col['Operating Time'] = np.where(
    #    (df_col['datetime'].dt.hour > 6) &
    #    (df_col['datetime'].dt.hour < 20) &
    #    (df_col['weekday'] == True), 'Yes', 'No')
    #df_col['hour'] = df_col['datetime'].dt.hour
    return df_col

# HVAC Electricity = Heating electricity + Cooling electricity + Fans electricity + Pumps electricity

# df_2015 = get_ts_data(hdf, str_clm='3C', str_eff='High', str_yr='2015', str_run='run_2')
# df_2016 = get_ts_data(hdf, str_clm='3C', str_eff='High', str_yr='2016', str_run='run_2')

# df_sf_cols = pd.DataFrame(df_sf.columns).to_excel('df_sf_cols.xlsx')
# list_year = ['1997', '1998', '1999']
# k = 0
def get_data(clm, eff, year, occ):
    cooling_thermostat = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='ZoneThermostatCoolingSetpointTemperature')
    heating_thermostat = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='ZoneThermostatHeatingSetpointTemperature')
    occupancy = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='ZonePeopleOccupantCount')
    mean_temperature = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='ZoneMeanAirTemperature')
    mass_flow_rate = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='ZoneMechanicalVentilationMassFlowRate')
    vav_damper_pos = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='ZoneAirTerminalVAVDamperPosition')
    outdoor_T = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='SiteOutdoorAirWetbulbTemperature')
    day_index = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='SiteDayTypeIndex')
    radiation = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='SiteHorizontalInfraredRadiationRateperArea')
    humidity = get_df_from_hdf(hdf, climate=clm, efficiency=eff, year=year, str_run=occ, data_key='SiteOutdoorAirRelativeHumidity')

    # ____________________________________________IMPORT ONE YEAR DATASET FOR 1 ZONE________________________________________
    ht_thermostat = heating_thermostat['CONFROOM_BOT_1 ZN:Zone Thermostat Heating Setpoint Temperature[C]']
    cl_thermostat = cooling_thermostat['CONFROOM_BOT_1 ZN:Zone Thermostat Cooling Setpoint Temperature[C]']
    occup = occupancy['CONFROOM_BOT_1 ZN:Zone People Occupant Count[]']
    mfr = mass_flow_rate['CONFROOM_BOT_1 ZN:Zone Mechanical Ventilation Mass Flow Rate[kg/s]']
    mean_T = mean_temperature['CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]']
    vav_dp = vav_damper_pos['CONFROOM_BOT_1 ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
    t_out = outdoor_T['Environment:Site Outdoor Air Wetbulb Temperature[C]']
    day_ix = day_index['Environment:Site Day Type Index[]']
    rad = radiation['Environment:Site Horizontal Infrared Radiation Rate per Area[W/m2]']
    hum = humidity['Environment:Site Outdoor Air Relative Humidity[%]']

    df = pd.concat([ht_thermostat, cl_thermostat, occup, mfr, vav_dp, t_out, day_ix, rad, hum, mean_T], axis=1)
    return df



clm = '3C'
eff = 'High'
occ = 'run_1'
zone = 'CONFROOM_BOT_1'

#df_2009 = get_data(clm=clm, eff=eff, year='2009', occ=occ)
#df_2010 = get_data(clm=clm, eff=eff, year='2010', occ=occ)
#df_2011 = get_data(clm=clm, eff=eff, year='2011', occ=occ)
#df_2012 = get_data(clm=clm, eff=eff, year='2012', occ=occ)
df_2017 = get_data(clm=clm, eff=eff, year='2017', occ=occ)


#df_2005 = get_data(clm=clm, eff=eff, year='2005', occ=occ)
#df_2006 = get_data(clm=clm, eff=eff, year='2006', occ=occ)
#df_2007 = get_data(clm=clm, eff=eff, year='2007', occ=occ)
#df_2008 = get_data(clm=clm, eff=eff, year='2008', occ=occ)
#df_2009 = get_data(clm=clm, eff=eff, year='2009', occ=occ)


#df_2000 = get_data(clm=clm, eff=eff, year='2000', occ=occ)
#df_2001 = get_data(clm=clm, eff=eff, year='2001', occ=occ)
#df_2002 = get_data(clm=clm, eff=eff, year='2002', occ=occ)
#df_2003 = get_data(clm=clm, eff=eff, year='2003', occ=occ)
#df_2004 = get_data(clm=clm, eff=eff, year='2004', occ=occ)
df_TMY = get_data(clm=clm, eff=eff, year='TMY3', occ=occ)

#df_2006.to_csv('data/{}_{}_{}_2006_{}.csv'.format(zone,clm, eff, occ))
#df_2005.to_csv('data/{}_{}_{}_2005_{}.csv'.format(zone,clm, eff, occ))
#df_2007.to_csv('data/{}_{}_{}_2007_{}.csv'.format(zone,clm, eff, occ))
#df_2008.to_csv('data/{}_{}_{}_2008_{}.csv'.format(zone,clm, eff, occ))
#df_2009.to_csv('data/{}_{}_{}_2009_{}.csv'.format(zone,clm, eff, occ))

#df_2000.to_csv('data/{}_{}_{}_2000_{}.csv'.format(zone,clm, eff, occ))
#df_2001.to_csv('data/{}_{}_{}_2001_{}.csv'.format(zone,clm, eff, occ))
#df_2002.to_csv('data/{}_{}_{}_2002_{}.csv'.format(zone,clm, eff, occ))
#df_2003.to_csv('data/{}_{}_{}_2003_{}.csv'.format(zone,clm, eff, occ))
#df_2004.to_csv('data/{}_{}_{}_2004_{}.csv'.format(zone,clm, eff, occ))
df_TMY.to_csv('data/{}_{}_{}_TMY3_{}.csv'.format(zone,clm, eff, occ))

#df_2013.to_csv('data/{}_{}_2013_{}.csv'.format(clm, eff, occ))
#df_2014.to_csv('data/{}_{}_2014_{}.csv'.format(clm, eff, occ))
#df_2015.to_csv('data/{}_{}_2015_{}.csv'.format(clm, eff, occ))
#df_2016.to_csv('data/{}_{}_2016_{}.csv'.format(clm, eff, occ))
df_2017.to_csv('data/{}_{}_{}_2017_{}.csv'.format(zone,clm, eff, occ))




# # Plotting
# plt.plot(cooling_thermostat['CONFROOM_BOT_1 ZN:Zone Thermostat Cooling Setpoint Temperature[C]'], label='cooling')
# plt.plot(heating_thermostat['CONFROOM_BOT_1 ZN:Zone Thermostat Heating Setpoint Temperature[C]'], label='heating')
# plt.plot(occupancy['CONFROOM_BOT_1 ZN:Zone People Occupant Count[]'], label='occupancy')
# plt.plot(mean_temperature['CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]'], label='mean_T')
# plt.plot(mass_flow_rate['CONFROOM_BOT_1 ZN:Zone Mechanical Ventilation Mass Flow Rate[kg/s]']*100, label='mass flow rate')
# plt.legend()
# plt.xlim(200, 250)
# plt.show()



# TODO: create a dataset for each year about San Francisco and than concatenate all of them to create a
#  unique dataset for standard efficiency.
"""
# in str_eff non posso mettere un int ma una stringa, quindi creo una lista di stringhe con tutti gli anni

list_years = ['1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
#list_years = ['1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']

#'1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',

df_tot = pd.DataFrame()
df = pd.DataFrame()
for year in list_years:
    df = get_ts_data(hdf, str_clm='3C', str_eff='Standard', str_yr=year, str_run='run_2') # non prende dal 1980 al 1996
    df_tot = pd.concat([df_tot, df], axis=0) #, columns=cols
    #df_tot = df_tot.append(df)
    #df = pd.DataFrame()

df_tot.to_csv('df_tot_from_2000_to_2009.csv')

"""