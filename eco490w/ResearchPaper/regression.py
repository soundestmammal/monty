import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Data_V5.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset[['PMODE', 'HHSIZE', 'DNYC', 'TOTTR_R', 'INCOM_R', 'HHVEH_R', 'GENDER', 'AGE_R', 'LIC' ,'TRPDUR', 'TRPDIST_HN']]
datasetv8 = dataset.drop(['OZIP', 'TOTTR', 'LTMODE_AGG', 'DTPURP', 'DTPURP2', 'ORIG_HOME', 'DEST_HOME' ], axis=1)

dataset.to_csv('Data_V51.csv', sep=',')

datasetv8_1 = pd.read_csv('DATA_V2-8.csv')

datasetv8_1 = datasetv8_1.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
datasetv8_1_anon = datasetv8_1.drop(['PLSAM', 'SAMPN', 'PERNO', 'PLANO'], axis=1)
dataset_anon = datasetv8_1_anon


dataset_anon.to_csv('Data_V3-0.csv', sep=',')

ordered = pd.read_csv('Data_V3-2.csv')

pipe_test = dataset_anon[dataset_anon.PMODE != 2]

pipe_test = pipe_test[pipe_test.PMODE != 26]
pipe_test.groupby('').size()

pipe_test.to_csv('Data_V3-1.csv', sep=',')


rearrange_test = pipe_test[['PMODE', 'TOTTR_R', 'INCOM_R', 'HHVEH_R', 'DNYC', 'HHSIZE', 'TRPDUR', 'TRPDIST_HN']]
rearrange_test.to_csv('Data_V3-2.csv', sep=',')


dataset = dataset.drop(['TLFC1'], axis=1)


clean_dataset = dataset.drop([ 'TOUR_PURP','DMPO','OMPO','ONYC','HMPO','HNYC','HSTATE','HCITY_MCD','HCOUNTY' ,'WHT_FAC3_VOCC','WHT_FAC3','TOURFAC','HH_WHT2', 'HH1','TRIPDIST','ACTDUR','SERVC','HHMEM','DTYPE', 'TOUR_ID', 'SUBT_ID', 'PERTYPE', 'PER1','GTYPE', 'HTAZ', 'HTRACT', 'OTAZ', 'DTAZ', 'NO_TAZ', 'PLOC', 'PRKTY', 'PAYPK', 'PKAMT', 'PKUNT', 'TOLFT', 'TLONB', 'TLFC1', 'TOPN1', 'TOLE1', 'TOLX1', 'TLLC1', 'TLFR1', 'TLFC2', 'TOPN2', 'TOLE2', 'TOLX2', 'TLLC2', 'TLFR2', 'TLFC3', 'TOPN3', 'TOLE3', 'TOLX3', 'TLLC3', 'TLFR3', 'O_TLFR3', 'ROUTE', 'FARE', 'FAREC', 'BUSPS', 'MTABP', 'BPFAR', 'FRBAS', 'TRP_DEP_HR', 'TRP_DEP_MIN', 'TRP_ARR_HR', 'TRP_ARR_MIN', 'OTRACT', 'DTRACT', 'OMCD', 'DMCD', 'UNIQUEID2', 'UNIQUEID3', 'UNIQUEID4', 'HHSIZ_R', 'TOD_PEAK', 'ODTPURP2_R', 'ODTPURP2', 'ODTPURP1', 'ODTPURP', 'DTPURP_R', 'DCOUNTY', 'OCOUNTY', 'ADJ_COUNTY', 'GEO_GROUP3_D', 'GEO_GROUP2_D', 'GEO_GROUP1_D', 'GEO_GROUP1_O', 'GEO_GROUP3', 'GEO_GROUP2', 'LTRIPNO', 'TOUR_ID', 'SUBT_ID', 'DTYPE', 'DOW', 'PER2', 'PER3', 'PER4', 'PER5', 'NONHH', 'VEHNO', 'DYGOV', 'PLOC' ], axis=1)
clean_dataset.to_csv('Data_V2-4.csv', sep=',')
new_set = pd.read_csv('Data_V2-6.csv')

new_set = new_set[new_set.GEO_GROUP1 == 3]
new_set = new_set[new_set.TOUR_PURP == 1]
new_set = new_set[new_set.GEO_GROUP2_O == 2]
new_set = new_set.drop(['PMODE1', 'PAMODE', 'PMODE_R', 'PMODE_R2', 'PMODE_R3', 'WORKTRIP'], axis=1)

new_set.to_csv('Data_V2-7.csv', sep=',')


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

# Data V2-7 REmove
new_set = new_set.drop(['PMODE1', 'PAMODE', 'PMODE_R', 'PMODE_R2', 'PMODE_R3', 'WORKTRIP'], axis=1)
new_set = new_set.drop(['GEO_GROUP2_O', 'GEO_GROUP3_O'], axis=1)
new_set = new_set.drop(['TRIPDIST_R1', 'TRIPDIST_R2', 'TRPDUR_R', 'TOD_R1', 'TOD_R'], axis=1)



pipe_test.groupby('PMODE').size()

# AfterI drop all of the unwanted data

# 1. I want to begin all my trips from home

# 2. I want to create a seperate set for just Long Island Reisdents

# 3. I want to see if the sample size ia big enough


person = pd.read_excel('PER_Public.xlsx')
person = person.filter(items=['PSAMP', 'SAMPN', 'PERNO', 'GENDER', 'AGE_R', 'LIC'])

# Dataset with people information
dataset = pd.read_csv('DATA_V2-8.csv')

sampn = dataset.iloc[:, 5].values

ndataset = dataset.assign(identifier = dataset['SAMPN'] + dataset['PERNO'])  


upPerson = person.assign(identifier = person['SAMPN'] + person['PERNO'])
droppedPerson = upPerson.drop_duplicates(subset=['identifier'])

dup = tryPerson.duplicated('identifier')
tryPerson[-1] = dup
tryPerson[-1].unique()

dup.value_counts()
    
identifier = []

for row in person:
    sampn = person['SAMPN']

readyMerge = droppedPerson.drop(['PSAMP', 'SAMPN', 'PERNO'], axis=1)


# merge ndataset and droppedPerson

hope = pd.merge(ndataset,readyMerge)

hopeDrop = hope.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0'], axis=1)
hope_anon = hopeDrop.drop(['PLSAM', 'SAMPN', 'PERNO', 'PLANO', 'identifier'], axis=1)
dataset_anon = datasetv8_1_anon

dataseter = pd.read_csv('Data_V3-3.csv')
dataseter = dataseter.drop(['Unnamed: 0'], axis=1)

dataseter.to_csv('Data_V4.csv', sep=',')
hope_anon.to_csv('Data_V5.csv', sep=',')

dataseter[0] = dataseter['PMODE']
hope_anon[0] = hope_anon['PMODE']

hope_anon = hope_anon.drop([], axis=1)

