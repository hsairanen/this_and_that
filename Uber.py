
#%%

import os
import pandas as pd
from scipy.stats import ttest_ind

wd = os.getcwd()

#%%

# Read data
df = pd.read_excel(wd+"\\Innovation_at_Uber.xlsx", sheet_name=2)

# Control group
cg = df[df['treat']==False].copy()
# Treat group
tg = df[df['treat']==True].copy()

cg['trips']=cg['trips_pool']+cg['trips_express']
tg['trips']=tg['trips_pool']+tg['trips_express']

#%%

# PROBLEM 1

#1. Do commuting hours experience a higher number of ridesharing (Express + POOL) trips
#compared to non-commuting hours? 

cg_non_com = cg.loc[cg['commute']==False,'trips']
cg_com = cg.loc[cg['commute']==True,'trips']

non_com_trips = cg.loc[cg['commute']==False,'trips'].sum()
com_trips = cg.loc[cg['commute']==True,'trips'].sum()

print(f"Non-commuting trips: {non_com_trips}")
print(f"Commuting trips: {com_trips}")

# %%

#2. What is the difference in the number of ridesharing trips between commuting and noncommuting hours?

print(f"The difference: {non_com_trips-com_trips}")

# %%

#3. Is the difference statistically significant at the 5% confidence level?

# Welch's two-sample t-test
t_stat, p_value = ttest_ind(cg_com, cg_non_com, equal_var=False, nan_policy="omit")

print(f"Welch's two-sample t-test statistic {t_stat:.4f} and p-value {p_value:.6f}")

# %%

# 4. Do riders use Express at higher rates during commuting hours compared to non-commuting
#hours? 

cg_express_non_com = cg.loc[cg['commute']==False,'trips_express']
cg_express_com = cg.loc[cg['commute']==True,'trips_express']

non_com_express = cg.loc[cg['commute']==False,'trips_express'].sum()
com_express = cg.loc[cg['commute']==True,'trips_express'].sum()

print(f"Non-commuting Express trips: {non_com_express}")
print(f"Commuting Express trips: {com_express}")

#%%

# 5. What is the difference in the share of Express trips between commuting and non-commuting
# hours? 

non_com_express_share = non_com_express / non_com_trips
com_express_share = com_express / com_trips

print(f"Non commuting share: {non_com_express_share:.4f}")
print(f"Commuting share: {com_express_share:.4f}")

print(f"Difference in Express share: {com_express_share - non_com_express_share:.4f}")

#%%

#6. Is the difference statistically significant at the 5% confidence level?

# Welch's two-sample t-test for Express share difference
t_stat, p_value = ttest_ind(cg_express_com, cg_express_non_com, equal_var=False, nan_policy="omit")

print(f"Test statistic {t_stat:.4f} and p-value {p_value:.7f}")

# %%
# 7. Assume that riders pay $12.5 on average for a POOL ride, and $10 for an Express ride. What is
# the difference in revenues between commuting and non-commuting hours?

# POOL trips for commuting and non-commuting hours
non_com_pool = cg.loc[cg['commute']==False,'trips_pool'].sum()
com_pool = cg.loc[cg['commute']==True,'trips_pool'].sum()

# Revenues
non_com_revenue = non_com_pool * 12.5 + non_com_express * 10
com_revenue = com_pool * 12.5 + com_express * 10

print(f"Non-commuting revenue: ${non_com_revenue:.2f}")
print(f"Commuting revenue: ${com_revenue:.2f}")
print(f"Difference in revenue: ${com_revenue - non_com_revenue:.2f}")

#%%

# 8. Is the difference statistically significant at the 5% confidence level?
    
# Revenue for each observation
cg_revenue_non_com = cg.loc[cg['commute']==False,'trips_pool'] * 12.5 + cg.loc[cg['commute']==False,'trips_express'] * 10
cg_revenue_com = cg.loc[cg['commute']==True,'trips_pool'] * 12.5 + cg.loc[cg['commute']==True,'trips_express'] * 10

# Welch's two-sample t-test for revenue difference
t_stat, p_value = ttest_ind(cg_revenue_com, cg_revenue_non_com, equal_var=False, nan_policy="omit")

print(f"Test statistic {t_stat:.4f} and p-value {p_value:.7f}")

# %%

# 9. Assume again that riders pay $12.5 on average for a POOL ride, and $10 for an Express ride.
# What is the difference in profits per trip between commuting and non-commuting hours?

# Profit per trip for commuting and non-commuting hours
non_com_profit_per_trip = non_com_revenue / non_com_trips
com_profit_per_trip = com_revenue / com_trips

print(f"Non-commuting profit per trip: ${non_com_profit_per_trip:.2f}")
print(f"Commuting profit per trip: ${com_profit_per_trip:.2f}")
print(f"Difference in profit per trip: ${com_profit_per_trip - non_com_profit_per_trip:.2f}")

#%%

# 10. Is the difference statistically significant at the 5% confidence level?

# Profit per trip for each observation
cg_profit_per_trip_non_com = (cg.loc[cg['commute']==False,'trips_pool'] * 12.5 + cg.loc[cg['commute']==False,'trips_express'] * 10) / (cg.loc[cg['commute']==False,'trips_pool'] + cg.loc[cg['commute']==False,'trips_express'])
cg_profit_per_trip_com = (cg.loc[cg['commute']==True,'trips_pool'] * 12.5 + cg.loc[cg['commute']==True,'trips_express'] * 10) / (cg.loc[cg['commute']==True,'trips_pool'] + cg.loc[cg['commute']==True,'trips_express'])

# Welch's two-sample t-test for profit per trip difference
t_stat, p_value = ttest_ind(cg_profit_per_trip_com, cg_profit_per_trip_non_com, equal_var=False, nan_policy="omit")

print(f"Test statistic {t_stat:.4f} and p-value {p_value:.4f}")

# %%

# PROBLEM 2

def do_analysis(is_commute):

    # 1. What is the difference in the number of ridesharing trips between the treatment and control 
    # groups during commuting hours? 

    # Treatment and commuting group commuting trips 
    tg_com_trips = tg.loc[tg['commute']==is_commute,'trips']
    cg_com_trips = cg.loc[cg['commute']==is_commute,'trips']

    tg_com_trips_total = tg_com_trips.sum()
    cg_com_trips_total = cg_com_trips.sum() 

    print(f"Treatment: {tg_com_trips_total}")
    print(f"Control: {cg_com_trips_total}")
    print(f"Difference in the number of ridesharing trips: {tg_com_trips_total - cg_com_trips_total}")

    # 2. Is the difference statistically significant at the 5% confidence level?

    t_stat, p_value = ttest_ind(tg_com_trips, cg_com_trips, equal_var=False, nan_policy="omit")
    print(f"Test statistic {t_stat:.4f} and p-value {p_value:.4f}")

    # 3. What is the difference in the number of rider cancellations between the treatment and control 
    # groups during commuting hours? 
    tg_com_cancellations = tg.loc[tg['commute']==is_commute,'rider_cancellations']
    cg_com_cancellations = cg.loc[cg['commute']==is_commute,'rider_cancellations']

    tg_com_cancellations_total = tg_com_cancellations.sum()
    cg_com_cancellations_total = cg_com_cancellations.sum()

    print(f"Treatment: {tg_com_cancellations_total}")
    print(f"Control: {cg_com_cancellations_total}")
    print(f"Difference in the number of rider cancellations: {tg_com_cancellations_total - cg_com_cancellations_total}")
    
    # 4. Is the difference statistically significant at the 5% confidence level? 

    t_stat, p_value = ttest_ind(tg_com_cancellations, cg_com_cancellations, equal_var=False, nan_policy="omit")
    print(f"Test statistic {t_stat:.4f} and p-value {p_value:.4f}")

    # 5. What is the difference in driver payout per trip between the treatment and control groups 
    # during commuting hours?

    tg_com_driver_payout = tg.loc[tg['commute']==is_commute,'total_driver_payout']
    cg_com_driver_payout = cg.loc[cg['commute']==is_commute,'total_driver_payout']

    tg_com_driver_payout_per_trip = tg_com_driver_payout.sum() / tg_com_trips_total
    cg_com_driver_payout_per_trip = cg_com_driver_payout.sum() / cg_com_trips_total

    print(f"Treatment: {tg_com_driver_payout_per_trip}")
    print(f"Control: {cg_com_driver_payout_per_trip}")
    print(f"Difference in driver payout per trip: ${tg_com_driver_payout_per_trip - cg_com_driver_payout_per_trip:.2f}")

    # 6. Is the difference statistically significant at the 5% confidence level? 
    
    tg_com_driver_payout_per_trip = tg.loc[tg['commute']==is_commute,'total_driver_payout'] / tg.loc[tg['commute']==True,'trips']
    cg_com_driver_payout_per_trip = cg.loc[cg['commute']==is_commute,'total_driver_payout'] / cg.loc[cg['commute']==True,'trips']

    t_stat, p_value = ttest_ind(tg_com_driver_payout_per_trip, cg_com_driver_payout_per_trip, equal_var=False, nan_policy="omit")

    print(f"Test statistic {t_stat:.4f} and p-value {p_value:.4f}")

    #7. What is the difference in overall match rate between the treatment and control groups during 
    # commuting hours? 

    tg_com_match_rate = tg.loc[tg['commute']==is_commute,'total_matches']
    cg_com_match_rate = cg.loc[cg['commute']==is_commute,'total_matches']

    tg_com_match_rate_avg = tg_com_match_rate.mean()
    cg_com_match_rate_avg = cg_com_match_rate.mean()

    print(f"Treatment: {tg_com_match_rate_avg}")
    print(f"Control: {cg_com_match_rate_avg}")
    print(f"Difference in overall match rate: {tg_com_match_rate_avg - cg_com_match_rate_avg:.4f}")

    # 8. Is the difference statistically significant at the 5% confidence level? 
    t_stat, p_value = ttest_ind(tg_com_match_rate, cg_com_match_rate, equal_var=False, nan_policy="omit")

    print(f"Test statistic {t_stat:.4f} and p-value {p_value:.4f}")

    # 9. What is the difference in double match rate between the treatment and control groups during 
    # commuting hours? 
        
    tg_com_double_match = tg.loc[tg['commute']==is_commute,'total_double_matches']
    cg_com_double_match = cg.loc[cg['commute']==is_commute,'total_double_matches']

    tg_com_double_match_avg = tg_com_double_match.mean()
    cg_com_double_match_avg = cg_com_double_match.mean()

    print(f"Treatment: {tg_com_double_match_avg}")
    print(f"Control: {cg_com_double_match_avg}")
    print(f"Difference in double match rate: {tg_com_double_match_avg - cg_com_double_match_avg:.4f}")

    # 10. Is the difference statistically significant at the 5% confidence level? 
    t_stat, p_value = ttest_ind(tg_com_double_match, cg_com_double_match, equal_var=False, nan_policy="omit")

    print(f"Test statistic {t_stat:.4f} and p-value {p_value:.4f}")

#%%

# For commuting hours
do_analysis(is_commute=True)

#%%

# For non-commuting hours
do_analysis(is_commute=False)

