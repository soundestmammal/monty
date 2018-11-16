import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('./Excel/LINKED_Public.xlsx')

dataset = dataset.drop(['TLFC1'], axis=1)


clean_dataset = dataset.drop([ 'TOUR_PURP','DMPO','OMPO','ONYC','HMPO','HNYC','HSTATE','HCITY_MCD','HCOUNTY' ,'WHT_FAC3_VOCC','WHT_FAC3','TOURFAC','HH_WHT2', 'HH1','TRIPDIST','ACTDUR','SERVC','HHMEM','DTYPE', 'TOUR_ID', 'SUBT_ID', 'PERTYPE', 'PER1','GTYPE', 'HTAZ', 'HTRACT', 'OTAZ', 'DTAZ', 'NO_TAZ', 'PLOC', 'PRKTY', 'PAYPK', 'PKAMT', 'PKUNT', 'TOLFT', 'TLONB', 'TLFC1', 'TOPN1', 'TOLE1', 'TOLX1', 'TLLC1', 'TLFR1', 'TLFC2', 'TOPN2', 'TOLE2', 'TOLX2', 'TLLC2', 'TLFR2', 'TLFC3', 'TOPN3', 'TOLE3', 'TOLX3', 'TLLC3', 'TLFR3', 'O_TLFR3', 'ROUTE', 'FARE', 'FAREC', 'BUSPS', 'MTABP', 'BPFAR', 'FRBAS', 'TRP_DEP_HR', 'TRP_DEP_MIN', 'TRP_ARR_HR', 'TRP_ARR_MIN', 'OTRACT', 'DTRACT', 'OMCD', 'DMCD', 'UNIQUEID2', 'UNIQUEID3', 'UNIQUEID4', 'HHSIZ_R', 'TOD_PEAK', 'ODTPURP2_R', 'ODTPURP2', 'ODTPURP1', 'ODTPURP', 'DTPURP_R', 'DCOUNTY', 'OCOUNTY', 'ADJ_COUNTY', 'GEO_GROUP3_D', 'GEO_GROUP2_D', 'GEO_GROUP1_D', 'GEO_GROUP1_O', 'GEO_GROUP3', 'GEO_GROUP2', 'LTRIPNO', 'TOUR_ID', 'SUBT_ID', 'DTYPE', 'DOW', 'PER2', 'PER3', 'PER4', 'PER5', 'NONHH', 'VEHNO', 'DYGOV', 'PLOC' ], axis=1)
clean_dataset.to_csv('Data_V2-4.csv', sep=',')
new_set = pd.read_csv('Data_V2-4.csv')

new_set = new_set[new_set.GEO_GROUP1 == 3]
new_set = new_set[new_set.TOUR_PURP == 1]
new_set = new_set[new_set.GEO_GROUP2_O == 2]

new_set.to_csv('Data_V2-6.csv', sep=',')

new_set.describe()

# Explanatory Variables that I am going to keep

#  Data_V2-2 Dropped a bunch of irrelevant variables
# Data_V2-3 Dropped a lot more variables got it down to 43
# Data V2-4 Dropped all observations of not in Geo_Group1 == 3 (Long Island)
# Now I have 19435 Observations
#Data V2-5 Only took observations where TourPurp is == 1
# This means that the tour purpose was going to work.
#Now that I eliminated them I went ahead and elminated from the dataset
# ALSO REMOVED the GEO_GROUP1

# DATA V2-6 Remove
# new_set = new_set.drop(['OTPURP', 'OTPURP_AGG', 'DTPURP_AGG', 'MODE_SAMP', 'TRPDUR_HN', 'DZIP', 'HZIP', 'WORK_PURP' ], axis=1)


# AfterI drop all of the unwanted data

# 1. I want to begin all my trips from home

# 2. I want to create a seperate set for just Long Island Reisdents

# 3. I want to see if the sample size ia big enough

