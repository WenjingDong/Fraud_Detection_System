pipeline_ONEHOT_ENCODED_FEATURES = [] # ['gender']
pipeline_LABEL_ENCODED_FEATURES =  ['cc_num',  'state', 'zip',  'merchant', 'category']# ['cc_num', 'street', 'city', 'state', 'zip', 'job', 'merchant', 'category']
# CATEGORICAL_FEATURES = ONEHOT_ENCODED_FEATURES + LABEL_ENCODED_FEATURES
# DATETIME_FEATURES = ["trans_year", "trans_month", "trans_day", "trans_hour", "trans_minute", "trans_second",  "age"] #, "time_interval", "time_interval_merchant", "rolling_count_24h", "rolling_amount_24h"]
# NUMERICAL_FEATURES = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
TIME_INTERVAL = 24 * 7

SCALED_FEATURES = ['cc_num_encoded', 'zip_encoded', 
                    'merchant_encoded', 'category_encoded', 'state_encoded'] \
                + ['amt', 'trans_month', 'trans_hour', 'age']

SELECTED_FEATURES = ['cc_num_encoded', 'zip_encoded', 
                    'merchant_encoded', 'category_encoded', 'state_encoded'] \
                + ['amt', 'trans_month', 'trans_hour', 'age']