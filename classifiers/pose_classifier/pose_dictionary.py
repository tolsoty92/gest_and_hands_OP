import joblib

# pose_dict = {
#                       ('ruki_vniz', 'ladoni_na_urovne_loktey',
#                        'lokti_vniz_ladon na urovne_plechey'):
#                           'Come on!',
#                       ('lokti_vniz_ladon na urovne_plechey',
#                        'ladoni_na_urovne_loktey',
#                        'lokti_vniz_ladon na urovne_plechey',
#                        'ladoni_na_urovne_loktey'):
#                           'Go away!'
#                         }

# joblib.dump(pose_dict, 'poses_combinatiob_dict')


d = joblib.load('poses_combinatiob_dict')
print(d[('ruki_vniz', 'ladoni_na_urovne_loktey', 'lokti_vniz_ladon na urovne_plechey')])