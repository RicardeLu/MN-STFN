ENV_CON_B = {
    'data_name':'TaxiBJ',
    'path':'./data/TaxiBJ/',
    'path_p':'./',
    'path_d':'./data/',
    'path_e':'./exp/',
    'flow_files_name':[
        'BJ2013_M32x32_T30_InOut.h5',
        'BJ2014_M32x32_T30_InOut.h5',
        'BJ2015_M32x32_T30_InOut.h5',
        'BJ2016_M32x32_T30_InOut.h5'
        ],
    'backgroud_file_name':'BJ_Meteorology.h5',
    'holiday_file':'BJ_Holiday.txt',
    'scale':1292,
    'read_cache': True
}

ENV_CON_N = {
    'data_name':'NYCBike',
    'path':'./data/NYCBike/',
    'path_p':'./',
    'path_d':'./data/',
    'path_e':'./exp/',
    'flow_files_name':['NYC14_M16x8_T60_NewEnd.h5'],
    'scale':267,
    'read_cache': True
}